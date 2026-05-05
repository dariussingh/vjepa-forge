from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from vjepa_forge.data.cache import recursive_to_device

def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _precision_dtype(precision: str) -> torch.dtype | None:
    lowered = str(precision).lower()
    if lowered == "bf16":
        return torch.bfloat16
    if lowered == "fp16":
        return torch.float16
    return None


@dataclass(frozen=True)
class RuntimeConfig:
    precision: str = "fp32"
    compile_enabled: bool = False
    compile_mode: str = "reduce-overhead"
    tf32: bool = True
    channels_last: bool = False
    sync_batchnorm: bool = False
    ddp_eval: bool = False


@dataclass
class RuntimeContext:
    requested_device: str
    config: RuntimeConfig
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_primary: bool
    amp_dtype: torch.dtype | None
    scaler: torch.cuda.amp.GradScaler | None

    @property
    def use_amp(self) -> bool:
        return self.device.type == "cuda" and self.amp_dtype is not None

    def autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def inference_context(self):
        return torch.inference_mode()

    def move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        moved = recursive_to_device(tensor, self.device)
        if isinstance(moved, torch.Tensor) and self.config.channels_last and moved.device.type == "cuda":
            if moved.ndim == 4:
                moved = moved.contiguous(memory_format=torch.channels_last)
            elif moved.ndim == 5:
                moved = moved.contiguous(memory_format=torch.channels_last_3d)
        return moved

    def prepare_module(self, module: torch.nn.Module, *, training: bool) -> torch.nn.Module:
        prepared = module.to(self.device)
        if self.config.sync_batchnorm and self.distributed and training:
            prepared = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prepared)
        if self.config.compile_enabled and self.device.type == "cuda" and hasattr(torch, "compile") and not self.distributed:
            prepared = torch.compile(prepared, mode=self.config.compile_mode)
        if self.distributed and training:
            prepared = DistributedDataParallel(
                prepared,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        return prepared


def normalize_runtime_config(data_cfg: dict[str, Any] | None) -> RuntimeConfig:
    payload = {} if data_cfg is None else dict(data_cfg.get("distributed", {}))
    precision = str(payload.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported distributed.precision: {precision}")
    return RuntimeConfig(
        precision=precision,
        compile_enabled=bool(payload.get("compile", False)),
        compile_mode=str(payload.get("compile_mode", "reduce-overhead")),
        tf32=_parse_bool(payload.get("tf32"), True),
        channels_last=bool(payload.get("channels_last", False)),
        sync_batchnorm=bool(payload.get("sync_batchnorm", False)),
        ddp_eval=bool(payload.get("ddp_eval", False)),
    )


def setup_runtime(*, device: str, data_cfg: dict[str, Any] | None = None) -> RuntimeContext:
    config = normalize_runtime_config(data_cfg)
    requested = str(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    wants_cuda = requested.startswith("cuda")
    distributed = world_size > 1 and wants_cuda and torch.cuda.is_available()
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if distributed:
        torch.cuda.set_device(local_rank)
        resolved_device = torch.device("cuda", local_rank)
    elif wants_cuda and torch.cuda.is_available():
        resolved_device = torch.device(requested)
    else:
        resolved_device = torch.device("cpu")
        distributed = False
        rank = 0
        local_rank = 0
        world_size = 1
    if resolved_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.tf32)
    amp_dtype = _precision_dtype(config.precision) if resolved_device.type == "cuda" else None
    scaler = None
    if resolved_device.type == "cuda" and config.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    return RuntimeContext(
        requested_device=requested,
        config=config,
        device=resolved_device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_primary=rank == 0,
        amp_dtype=amp_dtype,
        scaler=scaler,
    )


def distributed_sampler(dataset, *, shuffle: bool):
    if not dist.is_available() or not dist.is_initialized():
        return None
    return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)


def broadcast_object(value: Any, *, src: int = 0) -> Any:
    if not dist.is_available() or not dist.is_initialized():
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]
