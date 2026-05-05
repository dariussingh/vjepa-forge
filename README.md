# vjepa-forge

<p align="center">
  <strong>Train image and video downstream models with V-JEPA 2.1-style backbones from one CLI.</strong>
</p>

<p align="center">
  Classification, detection, segmentation, and anomaly workflows built around a single Forge dataset format.
</p>

<p align="center">
  <a href="#install">Install</a> |
  <a href="#supported-tasks">Supported Tasks</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#training">Training</a> |
  <a href="#validation">Validation</a> |
  <a href="#inference">Inference</a> |
  <a href="#dataset-conversion">Dataset Conversion</a> |
  <a href="#forge-datasets">Forge Datasets</a> |
  <a href="#python-api">Python API</a>
</p>

## Install

```bash
git clone git@github.com:dariussingh/vjepa-forge.git
cd vjepa-forge
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[detection,onnx,dev]
```

For GPU-side video decode, install the DALI extra or a matching DALI build for your CUDA version and use `data.video_backend=dali`. Without DALI, the runtime uses the built-in `decord` CPU path.

## Supported Tasks

| Task | Media | Support |
| --- | --- | --- |
| Classification | Image, Video | Train, val, predict |
| Detection | Image, Video | Train, val, predict |
| Segmentation | Image, Video | Train, val, predict |
| Anomaly | Image, Video | Train, val, predict, export |

## Getting Started

The new CLI is task-first:

```bash
forge classify train model=vjepa21-b.yaml data=kinetics400.yaml
forge detect train model=vjepa21-rfdetr.yaml data=coco.yaml
forge segment train model=vjepa21-vos.yaml data=davis.yaml
forge anomaly train model=vjepa21-predictor.yaml data=ucsd_ped2.yaml
```

You can keep datasets anywhere on disk. Each `data=...` argument points to a Forge dataset YAML, either:

- a shipped dataset config under `vjepa_forge/cfg/datasets/`
- your own `forge.yaml` path

By default, the active runtime resizes media to `384x384` for training, validation, prediction, and export. Override with `data.image_size=<size>` or `image_size=<size>` depending on your command.
Video datasets default to `data.video_backend=auto`, which uses DALI when available and otherwise falls back to `decord`.

## Training

Image classification:

```bash
forge classify train model=vjepa21-b.yaml data=/data/imagenet_forge/forge.yaml
```

Video classification:

```bash
forge classify train model=vjepa21-b.yaml data=/data/kinetics_forge/forge.yaml
```

Image detection:

```bash
forge detect train model=vjepa21-rfdetr.yaml data=/data/coco_forge/forge.yaml
```

Video detection:

```bash
forge detect train model=vjepa21-rfdetr.yaml data=/data/imagenet_vid_forge/forge.yaml
```

Cafe anomaly with the predictor-based V-JEPA 2.1 ViT-B path:

Convert Cafe into Forge format first:

```bash
forge convert cafe source=data/cafe out=data/cafe_forge task=anomaly media=video
```

The Cafe converter materializes trimmed `.mp4` clips under `data/cafe_forge/videos/` so each Forge record matches its own label interval.

```bash
forge anomaly train \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  train.epochs=10 \
  train.batch_size=16 \
  train.num_workers=8 \
  train.device=cuda
```

Overrides still use `key=value`:

```bash
forge classify train \
  model=vjepa21-b.yaml \
  data=/data/kinetics_forge/forge.yaml \
  train.epochs=10 \
  train.batch_size=4 \
  data.video_backend=auto \
  train.device=cpu
```

## Validation

```bash
forge classify val model=vjepa21-b.yaml data=/data/imagenet_forge/forge.yaml
forge detect val model=vjepa21-rfdetr.yaml data=/data/coco_forge/forge.yaml
```

Cafe anomaly validation:

```bash
forge anomaly val \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  train.device=cuda
```

The current Cafe converter writes the same held-out clips to both `val` and `test`.
Anomaly checkpoints and reports are written under `outputs/vjepa-forge/anomaly/cafe_forge/`.
For video tasks, decoded clip loading is shared across tasks and can be tuned with `train.num_workers`, `train.prefetch_factor`, `train.persistent_workers`, and `train.reader_cache_size`.
To force GPU-side decode, set `data.video_backend=dali`. The DALI path uses DALI experimental video decode APIs so anomaly sliding windows and other nonzero-offset clip reads work through the same backend.

> **Device note:** `train.device` is the global device setting for all modes (train, val, predict, export). There is no separate `val.device` — pass `train.device=cuda` in every command that needs GPU.

## Inference

```bash
forge classify predict model=vjepa21-b.yaml data=/data/imagenet_forge/forge.yaml
forge anomaly predict model=vjepa21-predictor.yaml data=/data/ucsd_forge/forge.yaml
```

Cafe anomaly prediction on the held-out split:

```bash
forge anomaly predict \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  train.device=cuda
```

## Export

Export is currently implemented for the **anomaly task only**. Anomaly export produces ONNX:

```bash
forge anomaly export \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  export.output_path=weights/cafe_anomaly_vitb.onnx
```

## Performance

Three levers that significantly reduce training time with no accuracy cost:

| Setting | Effect |
|---|---|
| `distributed.precision=bf16` | AMP bf16 mixed-precision — typically 1.5–2× faster on Ampere+ |
| `distributed.compile=true` | `torch.compile` on the frozen feature extractor — ~20–40% backbone throughput gain |
| `data.video_backend=dali` | GPU-side video decode — eliminates CPU decode bottleneck for large video datasets |
| `data.feature_cache=true` | Pre-extract and disk-cache backbone features — eliminates backbone forward pass from the training loop entirely |

For anomaly training on large video datasets, combining `feature_cache=true` with `feature_cache_dtype=fp16` reduces both disk usage and cache-build time by ~50% vs fp32.

## Config Reference

Key config sections and their supported keys (for `load_runtime_config` / `forge anomaly train` style commands):

**`train.*`**: `epochs`, `batch_size`, `lr`, `lr_mode`, `lr_scale_rule`, `reference_batch_size`, `reference_lr`, `weight_decay`, `device`, `num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`, `reader_cache_size`, `seed`, `resume`, `project`, `name`, `exist_ok`, `save`, `save_period`, `scheduler`, `early_stopping`

**`data.*`**: `image_size`, `image_backend`, `video_backend`, `feature_cache`, `feature_cache_root`, `feature_cache_build_on_miss`, `feature_cache_readonly`, `feature_cache_shard_size`, `feature_cache_dtype`, `train_fraction`, `past_frames`, `future_frames`, `stride`, `augment`

**`val.*`**: `batch_size`, `num_workers`, `split`, `checkpoint_target`, `checkpoint_path`, `threshold_std_multiplier`, `smoothing_window`

**`distributed.*`**: `precision` (`fp32`/`bf16`/`fp16`), `compile`, `compile_mode`, `backend`, `strategy`, `sync_batchnorm`, `tf32`, `channels_last`

## Dataset Conversion

External dataset formats should be converted into the canonical Forge layout before training.

Examples:

```bash
# Only the cafe converter is currently implemented:
forge convert cafe source=data/cafe out=data/cafe_forge task=anomaly media=video
```

> **Note:** Converters for COCO, Kinetics, DAVIS, and UCSD are not yet implemented. Use the Forge dataset layout directly for those formats.

## Forge Datasets

Each dataset split is either `media: image` or `media: video`.

Canonical layout:

```text
dataset/
  forge.yaml
  images/
    train/
    val/
  videos/
    train/
    val/
  labels/
    train/
    val/
  masks/
    train/
    val/
  splits/
    train.txt
    val.txt
    test.txt
```

Example `forge.yaml`:

```yaml
path: /data/my_dataset

task: detect
media: image

names:
  0: person
  1: car

splits:
  train: splits/train.txt
  val: splits/val.txt
  test: splits/test.txt

labels:
  format: forge-yolo
  root: labels

masks:
  root: masks
```

Image split files contain media-relative paths such as:

```text
images/train/000001.jpg
images/train/000002.jpg
```

Video split files contain media-relative paths such as:

```text
videos/train/clip001.mp4
videos/train/clip002.mp4
```

Matching labels live under `labels/<split>/` with the same stem:

```text
images/train/000001.jpg -> labels/train/000001.txt
videos/train/clip001.mp4 -> labels/train/clip001.txt
```

## Label Format

Classification:

```text
cls <class_id>
cls <class_id> <start_frame> <end_frame>
```

Detection:

```text
det <class_id> <x_center> <y_center> <width> <height>
det <frame_idx> <class_id> <x_center> <y_center> <width> <height>
```

Segmentation:

```text
seg <class_id> <x1> <y1> ... <xn> <yn>
seg <frame_idx> <class_id> <x1> <y1> ... <xn> <yn>
```

Anomaly:

```text
ano normal
ano abnormal <class_id>
ano abnormal <start_frame> <end_frame> <class_id>
```

The runtime parser also supports `ano_box` and `ano_seg` for optional spatial anomaly supervision.

## Python API

```python
from vjepa_forge import ForgeModel

model = ForgeModel(
    "vjepa21-rfdetr.yaml",
    data={"task": "detect", "media": "image", "image_size": 64},
)
```

## License

This repository is released under the MIT License.

Some copied or adapted upstream components are covered by their original notices. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
