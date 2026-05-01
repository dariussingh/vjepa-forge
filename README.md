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

The Cafe converter materializes trimmed clip tensors under `data/cafe_forge/videos/` so each Forge record matches its own label interval.

```bash
forge anomaly train \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  train.epochs=10 \
  train.batch_size=1 \
  train.device=cuda
```

Overrides still use `key=value`:

```bash
forge classify train \
  model=vjepa21-b.yaml \
  data=/data/kinetics_forge/forge.yaml \
  train.epochs=10 \
  train.batch_size=4 \
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

Anomaly export currently produces ONNX on the active `forge` path:

```bash
forge anomaly export \
  model=vjepa21-predictor.yaml \
  data=data/cafe_forge/forge.yaml \
  model.backbone.checkpoint=weights/vjepa2_1_vitb_dist_vitG_384.pt \
  data.image_size=384 \
  export.output_path=weights/cafe_anomaly_vitb.onnx
```

## Dataset Conversion

External dataset formats should be converted into the canonical Forge layout before training.

Examples:

```bash
forge convert coco source=/data/coco out=/data/coco_forge task=detect media=image
forge convert kinetics source=/data/kinetics out=/data/kinetics_forge task=classify media=video
forge convert davis source=/data/DAVIS out=/data/davis_forge task=segment media=video
forge convert ucsd source=/data/UCSDped2 out=/data/ucsd_forge task=anomaly media=video
forge convert cafe source=data/cafe out=data/cafe_forge task=anomaly media=video
```

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
