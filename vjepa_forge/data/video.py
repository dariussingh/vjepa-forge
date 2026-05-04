from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_READER_CACHE_BY_WORKER: dict[tuple[int, int | None], OrderedDict[str, Any]] = {}
_FRAME_COUNT_CACHE_BY_WORKER: dict[tuple[int, int | None], dict[str, int]] = {}
_ENCODED_VIDEO_CACHE_BY_WORKER: dict[tuple[int, int | None], OrderedDict[str, np.ndarray]] = {}
_DALI_PIPELINE_CACHE_BY_WORKER: dict[tuple[int, int | None], OrderedDict[tuple[int, int], Any]] = {}
_DEFAULT_READER_CACHE_SIZE = 4
_DEFAULT_VIDEO_BACKEND = "auto"


def _worker_key() -> tuple[int, int | None]:
    worker_info = torch.utils.data.get_worker_info()
    worker_id = None if worker_info is None else int(worker_info.id)
    return (os.getpid(), worker_id)


def _read_tensor_video(path: Path) -> torch.Tensor:
    tensor = torch.load(path)
    if tensor.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W] tensor video at {path}, got {tuple(tensor.shape)}")
    return tensor.float()


def _reader_cache(cache_size: int) -> OrderedDict[str, Any]:
    key = _worker_key()
    cache = _READER_CACHE_BY_WORKER.setdefault(key, OrderedDict())
    while len(cache) > max(0, cache_size):
        cache.popitem(last=False)
    return cache


def _frame_count_cache() -> dict[str, int]:
    return _FRAME_COUNT_CACHE_BY_WORKER.setdefault(_worker_key(), {})


def _encoded_video_cache(cache_size: int) -> OrderedDict[str, np.ndarray]:
    key = _worker_key()
    cache = _ENCODED_VIDEO_CACHE_BY_WORKER.setdefault(key, OrderedDict())
    while len(cache) > max(0, cache_size):
        cache.popitem(last=False)
    return cache


def _dali_pipeline_cache(cache_size: int) -> OrderedDict[tuple[int, int], Any]:
    key = _worker_key()
    cache = _DALI_PIPELINE_CACHE_BY_WORKER.setdefault(key, OrderedDict())
    while len(cache) > max(0, cache_size):
        cache.popitem(last=False)
    return cache


def _has_dali() -> bool:
    try:
        import nvidia.dali  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_backend(
    *,
    source: Path,
    video_backend: str,
) -> str:
    requested = str(video_backend or _DEFAULT_VIDEO_BACKEND).lower()
    if requested not in {"auto", "decord", "dali"}:
        raise ValueError(f"Unsupported video backend: {video_backend}")
    if source.suffix == ".pt":
        return "tensor"
    if requested == "decord":
        return "decord"
    if requested == "dali":
        if not _has_dali():
            raise RuntimeError("data.video_backend=dali requested, but NVIDIA DALI is not installed")
        return "dali"
    if _has_dali():
        return "dali"
    return "decord"


def _get_video_reader(path: Path, *, reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE):
    try:
        import decord
    except Exception as exc:
        raise RuntimeError(f"Reading non-.pt video requires decord: {path}") from exc

    cache = _reader_cache(reader_cache_size)
    key = str(path.resolve())
    reader = cache.pop(key, None)
    if reader is None:
        reader = decord.VideoReader(str(path))
    cache[key] = reader
    while len(cache) > max(0, reader_cache_size):
        cache.popitem(last=False)
    return reader


def _get_encoded_video_bytes(path: Path, *, reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE) -> np.ndarray:
    cache = _encoded_video_cache(reader_cache_size)
    key = str(path.resolve())
    encoded = cache.pop(key, None)
    if encoded is None:
        encoded = np.fromfile(path, dtype=np.uint8)
    cache[key] = encoded
    while len(cache) > max(0, reader_cache_size):
        cache.popitem(last=False)
    return encoded


class _DaliVideoPipeline:
    def __init__(self, *, image_size: int, device_id: int) -> None:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn

        @pipeline_def(
            batch_size=1,
            num_threads=2,
            device_id=device_id,
            exec_async=False,
            exec_pipelined=False,
        )
        def video_pipe():
            encoded = fn.external_source(name="encoded", device="cpu")
            start_frame = fn.external_source(name="start_frame", device="cpu")
            sequence_length = fn.external_source(name="sequence_length", device="cpu")
            stride = fn.external_source(name="stride", device="cpu")
            frames = fn.experimental.decoders.video(
                encoded,
                device="mixed",
                start_frame=start_frame,
                sequence_length=sequence_length,
                stride=stride,
                build_index=True,
            )
            return fn.resize(frames, device="gpu", resize_x=int(image_size), resize_y=int(image_size))

        self.pipeline = video_pipe()
        self.pipeline.build()

    def run(self, *, encoded: np.ndarray, start_frame: int, sequence_length: int, stride: int) -> Any:
        self.pipeline.feed_input("encoded", [encoded])
        self.pipeline.feed_input("start_frame", np.asarray([start_frame], dtype=np.int32))
        self.pipeline.feed_input("sequence_length", np.asarray([sequence_length], dtype=np.int32))
        self.pipeline.feed_input("stride", np.asarray([stride], dtype=np.int32))
        self.pipeline.reset()
        return self.pipeline.run()[0]


def _get_dali_pipeline(*, image_size: int, device_id: int, reader_cache_size: int) -> _DaliVideoPipeline:
    cache = _dali_pipeline_cache(reader_cache_size)
    key = (int(image_size), int(device_id))
    pipeline = cache.pop(key, None)
    if pipeline is None:
        pipeline = _DaliVideoPipeline(image_size=image_size, device_id=device_id)
    cache[key] = pipeline
    while len(cache) > max(0, reader_cache_size):
        cache.popitem(last=False)
    return pipeline


def _normalize_indices(total: int, *, clip_start: int, clip_len: int | None, stride: int) -> list[int]:
    if total <= 0:
        return []
    start = max(0, int(clip_start))
    step = max(1, int(stride))
    if clip_len is None:
        return list(range(start, total, step))
    indices = [min(start + (i * step), total - 1) for i in range(max(0, int(clip_len)))]
    return indices


def _pad_clip_tensor(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        return tensor
    if tensor.shape[0] >= target_len or tensor.shape[0] == 0:
        return tensor
    pad = tensor[-1:].repeat(target_len - tensor.shape[0], 1, 1, 1)
    return torch.cat([tensor, pad], dim=0)


def _read_video_clip_decord(
    source: Path,
    *,
    clip_start: int,
    clip_len: int | None,
    stride: int,
    image_size: int,
    reader_cache_size: int,
) -> torch.Tensor:
    reader = _get_video_reader(source, reader_cache_size=reader_cache_size)
    total = len(reader)
    indices = _normalize_indices(total, clip_start=clip_start, clip_len=clip_len, stride=stride)
    if not indices:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    frames = reader.get_batch(indices).asnumpy()
    sliced = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    sliced = _pad_clip_tensor(sliced, int(clip_len or 0))
    return F.interpolate(sliced, size=(image_size, image_size), mode="bilinear", align_corners=False)


def _read_video_clip_dali(
    source: Path,
    *,
    clip_start: int,
    clip_len: int | None,
    stride: int,
    image_size: int,
    reader_cache_size: int,
) -> torch.Tensor:
    if clip_len is None:
        frame_count = get_video_frame_count(source, reader_cache_size=reader_cache_size, video_backend="decord")
        stride_value = max(1, int(stride))
        clip_len = max(0, (frame_count - max(0, int(clip_start)) + stride_value - 1) // stride_value)
    target_len = int(clip_len)
    if target_len <= 0:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    encoded = _get_encoded_video_bytes(source, reader_cache_size=reader_cache_size)
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    pipeline = _get_dali_pipeline(
        image_size=image_size,
        device_id=device_id,
        reader_cache_size=reader_cache_size,
    )
    frames = pipeline.run(
        encoded=encoded,
        start_frame=max(0, int(clip_start)),
        sequence_length=target_len,
        stride=max(1, int(stride)),
    )
    if hasattr(frames, "as_tensor"):
        frames = frames.as_tensor()
    try:
        tensor = torch.utils.dlpack.from_dlpack(frames)
    except Exception:
        tensor = torch.from_numpy(frames.as_cpu().as_array())
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Unexpected DALI output shape: {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    tensor = tensor.permute(0, 3, 1, 2).contiguous() / 255.0
    return _pad_clip_tensor(tensor, target_len)


def read_video_clip(
    path: str | Path,
    *,
    clip_start: int = 0,
    clip_len: int | None = None,
    stride: int = 1,
    image_size: int = 384,
    reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE,
    video_backend: str = _DEFAULT_VIDEO_BACKEND,
) -> torch.Tensor:
    source = Path(path)
    backend = _resolve_backend(source=source, video_backend=video_backend)
    if backend == "tensor":
        video = _read_tensor_video(source)
        if clip_len is None:
            sliced = video[max(0, clip_start) :: max(1, stride)]
        else:
            end = clip_start + clip_len * stride
            sliced = video[max(0, clip_start) : end : max(1, stride)]
            sliced = _pad_clip_tensor(sliced, int(clip_len))
        if sliced.numel() > 0:
            sliced = F.interpolate(sliced, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return sliced
    if backend == "dali":
        return _read_video_clip_dali(
            source,
            clip_start=clip_start,
            clip_len=clip_len,
            stride=stride,
            image_size=image_size,
            reader_cache_size=reader_cache_size,
        )
    return _read_video_clip_decord(
        source,
        clip_start=clip_start,
        clip_len=clip_len,
        stride=stride,
        image_size=image_size,
        reader_cache_size=reader_cache_size,
    )


def get_video_frame_count(
    path: str | Path,
    *,
    reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE,
    video_backend: str = _DEFAULT_VIDEO_BACKEND,
) -> int:
    source = Path(path)
    if source.suffix == ".pt":
        return int(_read_tensor_video(source).shape[0])
    key = str(source.resolve())
    counts = _frame_count_cache()
    if key in counts:
        return counts[key]
    reader = _get_video_reader(source, reader_cache_size=reader_cache_size)
    counts[key] = int(len(reader))
    return counts[key]


def read_video_frames_uint8(
    path: str | Path,
    *,
    clip_start: int = 0,
    clip_len: int | None = None,
    stride: int = 1,
    reader_cache_size: int = _DEFAULT_READER_CACHE_SIZE,
) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".pt":
        tensor = _read_tensor_video(source)
        if clip_len is None:
            sliced = tensor[max(0, clip_start) :: max(1, stride)]
        else:
            end = clip_start + clip_len * stride
            sliced = tensor[max(0, clip_start) : end : max(1, stride)]
            sliced = _pad_clip_tensor(sliced, int(clip_len))
        if sliced.numel() == 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        if sliced.max() <= 1.0 and sliced.min() >= 0.0:
            scaled = (sliced.clamp(0.0, 1.0) * 255.0).round()
        else:
            scaled = sliced.clamp(0.0, 255.0)
        return scaled.byte().permute(0, 2, 3, 1).cpu().numpy()
    reader = _get_video_reader(source, reader_cache_size=reader_cache_size)
    total = len(reader)
    indices = _normalize_indices(total, clip_start=clip_start, clip_len=clip_len, stride=stride)
    if not indices:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    frames = reader.get_batch(indices).asnumpy()
    target_len = int(clip_len or 0)
    if target_len > 0 and frames.shape[0] < target_len and frames.shape[0] > 0:
        pad = np.repeat(frames[-1:, :, :, :], target_len - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames.astype(np.uint8, copy=False)
