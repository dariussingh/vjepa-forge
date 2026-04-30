# vjepa-forge

<p align="center">
  <strong>Forge image and video downstream models from V-JEPA 2.1.</strong>
</p>

<p align="center">
  Classification, detection, segmentation, and anomaly detection for image and video with a simple training and inference workflow.
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#install">Install</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#training">Training</a> |
  <a href="#validation">Validation</a> |
  <a href="#inference">Inference</a> |
  <a href="#export">Export</a> |
  <a href="#datasets">Datasets</a> |
  <a href="#recipes">Recipes</a>
</p>

---

> **vjepa-forge is an open-source framework for building image and video perception models with V-JEPA 2.1 backbones, dense predictive features, DETR/RF-DETR-style detection heads, and reproducible benchmark recipes.**

## Features

- Image classification
- Video classification
- Image detection
- Video detection
- Semantic segmentation
- Instance segmentation
- Video anomaly detection
- CLI for training, validation, inference, benchmarking, and export
- YAML-based recipe configuration

## Supported Tasks

| Task | Input | Commands |
| --- | --- | --- |
| Classification | Image, Video | `train`, `val`, `predict`, `export` |
| Detection | Image, Video | `train`, `val`, `predict` |
| Segmentation | Image | `train`, `val`, `predict` |
| Anomaly Detection | Video | `train`, `val`, `predict` |

## Install

Create a Python 3.11+ environment with PyTorch installed, then:

```bash
git clone git@github.com:dariussingh/vjepa-forge.git
cd vjepa-forge
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[detection,onnx]
```

## Getting Started

If you already keep datasets and checkpoints in a parent workspace:

```bash
ln -s ../data data
ln -s ../weights weights
```

Run a quick smoke benchmark:

```bash
vjepa-forge benchmark \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  data.image_size=64
```

## Training

### Image classification

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml
```

### Video classification

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/classification/kinetics400_vitb.yaml
```

### Image detection

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/detection/coco_rf_detr.yaml
```

### Video detection

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/detection/imagenet_vid_temporal_detr.yaml
```

### Semantic segmentation

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/segmentation/ade20k_semantic.yaml
```

### Instance segmentation

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/segmentation/coco_instance.yaml
```

### Anomaly detection

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/anomaly/ucsd_ped2_predictor.yaml
```

## Validation

```bash
vjepa-forge val \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml
```

```bash
vjepa-forge val \
  recipe=vjepa_forge/recipes/detection/coco_rf_detr.yaml
```

```bash
vjepa-forge val \
  recipe=vjepa_forge/recipes/detection/imagenet_vid_temporal_detr.yaml
```

## Inference

### PyTorch inference

```bash
vjepa-forge predict \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml
```

```bash
vjepa-forge predict \
  recipe=vjepa_forge/recipes/detection/coco_rf_detr.yaml
```

```bash
vjepa-forge predict \
  recipe=vjepa_forge/recipes/detection/imagenet_vid_temporal_detr.yaml
```

### ONNX Runtime inference

```bash
vjepa-forge predict \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  inference.backend=onnx \
  export.output_path=classification.onnx
```

## Export

### ONNX export

```bash
vjepa-forge export \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  export.output_path=classification.onnx
```

```bash
vjepa-forge export \
  recipe=vjepa_forge/recipes/classification/kinetics400_vitb.yaml \
  export.output_path=kinetics400.onnx
```

Detection, segmentation, and anomaly export support depends on the selected path and runtime backend. Video detection currently uses PyTorch inference.

## CLI Overrides

Override recipe values directly from the command line:

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/classification/kinetics400_vitb.yaml \
  data.image_size=224 \
  data.num_frames=8 \
  train.epochs=10
```

## Python API

```python
from vjepa_forge.configs.loader import load_recipe
from vjepa_forge.engine.trainer import build_model

cfg = load_recipe("vjepa_forge/recipes/classification/imagenet1k_vitb.yaml")
model = build_model(cfg)
```

## Datasets

`vjepa-forge` expects datasets under `data/` and checkpoints under `weights/`.

### ImageNet VID

Video detection expects this layout:

```text
data/imagenet_vid/
  Data/
    VID/
      train/
      val/
  Annotations/
    VID/
      train/
      val/
```

Helper script:

```bash
./scripts/download_imagenet_vid.sh
```

### Other dataset helpers

```bash
./scripts/download_coco_det.sh
./scripts/download_ucsd_ped2.sh
./scripts/download_cuhk_avenue.sh
./scripts/download_vjepa_weights.sh
```

## Recipes

Included recipes:

- `vjepa_forge/recipes/classification/imagenet1k_vitb.yaml`
- `vjepa_forge/recipes/classification/kinetics400_vitb.yaml`
- `vjepa_forge/recipes/detection/coco_rf_detr.yaml`
- `vjepa_forge/recipes/detection/imagenet_vid_temporal_detr.yaml`
- `vjepa_forge/recipes/segmentation/ade20k_semantic.yaml`
- `vjepa_forge/recipes/segmentation/coco_instance.yaml`
- `vjepa_forge/recipes/anomaly/ucsd_ped2_predictor.yaml`

## License

This repository is released under the MIT License.

Some copied or adapted upstream components are covered by their original notices. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
