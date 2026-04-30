# vjepa-forge

<p align="center">
  <strong>Forge image and video downstream models from V-JEPA 2.1.</strong>
</p>

<p align="center">
  Classification, detection, segmentation, and predictor-based anomaly detection with a unified V-JEPA-first training and inference interface.
</p>

<p align="center">
  <a href="#install">Install</a> |
  <a href="#usage">Usage</a> |
  <a href="#recipes">Recipes</a> |
  <a href="#export">Export</a> |
  <a href="#repo-layout">Repo Layout</a>
</p>

---

`vjepa-forge` is an open-source framework for building image and video perception models with V-JEPA 2.1 backbones, dense predictive features, DETR/RF-DETR-style detection heads, and reproducible benchmark recipes.

Core design rules:

- V-JEPA 2.1 is the default representation backbone.
- Images are treated as `T=1` videos at the public API layer.
- Recipes declare their grounding through a `references:` block.
- The CLI is intentionally Ultralytics-like: `train`, `val`, `predict`, `export`, and `benchmark`.

## Tasks

`vjepa-forge` is organized around four downstream task families:

| Task | Input | Current Path |
| --- | --- | --- |
| Classification | Image, Video | Unified local trainer and recipes |
| Detection | Image | Local RF-DETR-style implementation plus Ultralytics-oriented interfaces |
| Segmentation | Image, Video | Semantic, instance, and VOS-oriented heads |
| Anomaly Detection | Video | Predictor-based anomaly pipeline with JEPA-style feature prediction |

## Install

Create an environment with Python 3.11+ and PyTorch, then install the repo in editable mode:

```bash
cd /media/development/usi/vjepa2/vjepa-forge
python -m pip install -e .
```

Install optional extras when needed:

```bash
python -m pip install -e .[detection,onnx]
```

Notes:

- `detection` adds the packages used by the detection/evaluation path.
- `onnx` adds ONNX/ONNX Runtime support. Depending on your local PyTorch version, you may also need `onnxscript` for export.

## Usage

### CLI

The main interface is the `vjepa-forge` CLI:

```bash
vjepa-forge train recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml
vjepa-forge val recipe=vjepa_forge/recipes/detection/coco_rf_detr.yaml
vjepa-forge predict recipe=vjepa_forge/recipes/segmentation/ade20k_semantic.yaml source=example.jpg
vjepa-forge export recipe=vjepa_forge/recipes/anomaly/ucsd_ped2_predictor.yaml export.output_path=model.onnx
vjepa-forge benchmark recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml data.image_size=64
```

Override any config value with `key=value` pairs:

```bash
vjepa-forge train \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  data.image_size=64 \
  data.batch_size=1 \
  train.epochs=1
```

### Python

The same building blocks can be used directly from Python:

```python
from vjepa_forge.configs import load_recipe
from vjepa_forge.engine.trainer import build_model

cfg = load_recipe("vjepa_forge/recipes/classification/imagenet1k_vitb.yaml")
model = build_model(cfg)
```

Backbone access is also available directly:

```python
import torch

from vjepa_forge.backbones import VJEPAImageBackbone

backbone = VJEPAImageBackbone(name="vit_base", checkpoint=None)
x = torch.randn(1, 3, 384, 384)
features = backbone(x)
```

## Recipes

Current example recipes live under [`vjepa_forge/recipes`](./vjepa_forge/recipes):

- Classification:
  - `classification/imagenet1k_vitb.yaml`
  - `classification/kinetics400_vitb.yaml`
- Detection:
  - `detection/coco_rf_detr.yaml`
- Segmentation:
  - `segmentation/ade20k_semantic.yaml`
  - `segmentation/coco_instance.yaml`
- Anomaly:
  - `anomaly/ucsd_ped2_predictor.yaml`

Each recipe is expected to declare its grounding explicitly:

```yaml
references:
  backbone: [vjepa, vjepa2, vjepa2_1]
  dense_features: [dinov3]
  detection: [detr, rf_detr]
  training_interface: [ultralytics]
```

## Data

`vjepa-forge` expects repo-local `data/` and `weights/` directories. If you already have them in the parent workspace, a common setup is:

```bash
cd /media/development/usi/vjepa2/vjepa-forge
ln -s ../data data
ln -s ../weights weights
```

Current helper scripts are in [`scripts`](./scripts):

- `download_vjepa_weights.sh`
- `download_coco_det.sh`
- `download_ucsd_ped2.sh`
- `download_cuhk_avenue.sh`

At the moment, these are preparation stubs rather than full automated downloaders, so dataset placement should still be treated as an explicit local setup step.

## Export

ONNX export and ONNX Runtime inference are wired through the same config system:

```bash
vjepa-forge export \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  export.output_path=classification.onnx
```

```bash
vjepa-forge predict \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  inference.backend=onnx \
  export.output_path=classification.onnx
```

If export fails with a missing-module error, install the ONNX extras and `onnxscript` in the current environment.

## Repo Layout

```text
vjepa-forge/
  vjepa_forge/
    backbones/
    cli/
    configs/
    data/
    engine/
    export/
    heads/
    masks/
    metrics/
    models/
    predictors/
    recipes/
    tokenizers/
    utils/
  scripts/
  tests/
```

Important internal rule: all runtime code now lives directly under `vjepa_forge`. There is no `vendor/`, `src/`, `app/`, or `rfdetr/` bridge namespace in the repo runtime path.

## Status

What is in good shape now:

- V-JEPA 2.1 backbone wrappers for image and video
- local anomaly modeling/runtime path
- local RF-DETR-style detection path
- unified CLI/config loader
- ONNX export/inference hooks

What still needs iteration:

- richer real-dataset wiring across all tasks
- fuller benchmark coverage
- stronger docs beyond quickstart
- automated dataset download flows

## Development

Useful local checks:

```bash
python -m compileall vjepa_forge
```

```bash
vjepa-forge benchmark \
  recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml \
  data.image_size=64
```

## Citation

If you use `vjepa-forge`, cite the upstream representation work and downstream components that your recipe depends on, especially:

- V-JEPA / V-JEPA 2 / V-JEPA 2.1
- DINOv3
- DETR
- RF-DETR

