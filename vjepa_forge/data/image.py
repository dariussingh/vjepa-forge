from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


_ENCODED_IMAGE_CACHE_BY_WORKER: dict[tuple[int, int | None], OrderedDict[str, np.ndarray]] = {}
_DALI_IMAGE_PIPELINE_CACHE_BY_WORKER: dict[tuple[int, int | None], OrderedDict[tuple[int, int], Any]] = {}
_DEFAULT_IMAGE_BACKEND = "auto"
_DEFAULT_READER_CACHE_SIZE = 4


def _worker_key() -> tuple[int, int | None]:
    worker_info = torch.utils.data.get_worker_info()
    worker_id = None if worker_info is None else int(worker_info.id)
    return (os.getpid(), worker_id)


def _encoded_image_cache(cache_size: int) -> OrderedDict[str, np.ndarray]:
    key = _worker_key()
    cache = _ENCODED_IMAGE_CACHE_BY_WORKER.setdefault(key, OrderedDict())
    while len(cache) > max(0, cache_size):
        cache.popitem(last=False)
    return cache


def _dali_image_pipeline_cache(cache_size: int) -> OrderedDict[tuple[int, int], Any]:
    key = _worker_key()
    cache = _DALI_IMAGE_PIPELINE_CACHE_BY_WORKER.setdefault(key, OrderedDict())
    while len(cache) > max(0, cache_size):
        cache.popitem(last=False)
    return cache


def _has_dali() -> bool:
    try:
        import nvidia.dali  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_backend(*, image_backend: str) -> str:
    requested = str(image_backend or _DEFAULT_IMAGE_BACKEND).lower()
    if requested not in {"auto", "pil", "dali"}:
        raise ValueError(f"Unsupported image backend: {image_backend}")
    dali_ready = _has_dali() and torch.cuda.is_available()
    if requested == "pil":
        return "pil"
    if requested == "dali":
        if not dali_ready:
            raise RuntimeError("data.image_backend=dali requested, but NVIDIA DALI with CUDA is not available")
        return "dali"
    if dali_ready:
        return "dali"
    return "pil"


def _get_encoded_image_bytes(path: Path, *, reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE) -> np.ndarray:
    cache = _encoded_image_cache(reader_cache_size)
    key = str(path.resolve())
    encoded = cache.pop(key, None)
    if encoded is None:
        encoded = np.fromfile(path, dtype=np.uint8)
    cache[key] = encoded
    while len(cache) > max(0, reader_cache_size):
        cache.popitem(last=False)
    return encoded


class _DaliImagePipeline:
    def __init__(self, *, image_size: int, device_id: int) -> None:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types

        @pipeline_def(
            batch_size=1,
            num_threads=2,
            device_id=device_id,
            exec_async=False,
            exec_pipelined=False,
        )
        def image_pipe():
            encoded = fn.external_source(name="encoded", device="cpu")
            images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
            return fn.resize(images, device="gpu", resize_x=int(image_size), resize_y=int(image_size))

        self.pipeline = image_pipe()
        self.pipeline.build()

    def run(self, *, encoded: np.ndarray) -> Any:
        self.pipeline.feed_input("encoded", [encoded])
        self.pipeline.reset()
        return self.pipeline.run()[0]


def _get_dali_image_pipeline(*, image_size: int, device_id: int, reader_cache_size: int) -> _DaliImagePipeline:
    cache = _dali_image_pipeline_cache(reader_cache_size)
    key = (int(image_size), int(device_id))
    pipeline = cache.pop(key, None)
    if pipeline is None:
        pipeline = _DaliImagePipeline(image_size=image_size, device_id=device_id)
    cache[key] = pipeline
    while len(cache) > max(0, reader_cache_size):
        cache.popitem(last=False)
    return pipeline


def _read_image_pil(path: Path, *, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _read_image_dali(path: Path, *, image_size: int, reader_cache_size: int) -> torch.Tensor:
    encoded = _get_encoded_image_bytes(path, reader_cache_size=reader_cache_size)
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    pipeline = _get_dali_image_pipeline(image_size=image_size, device_id=device_id, reader_cache_size=reader_cache_size)
    image = pipeline.run(encoded=encoded)
    if hasattr(image, "as_tensor"):
        image = image.as_tensor()
    try:
        tensor = torch.utils.dlpack.from_dlpack(image)
    except Exception:
        tensor = torch.from_numpy(image.as_cpu().as_array())
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected DALI image output shape: {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.permute(2, 0, 1).contiguous() / 255.0


def read_image(
    path: str | Path,
    *,
    image_size: int = 640,
    image_backend: str = _DEFAULT_IMAGE_BACKEND,
    reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE,
) -> torch.Tensor:
    source = Path(path)
    backend = _resolve_backend(image_backend=image_backend)
    if backend == "dali":
        return _read_image_dali(source, image_size=image_size, reader_cache_size=reader_cache_size)
    return _read_image_pil(source, image_size=image_size)
