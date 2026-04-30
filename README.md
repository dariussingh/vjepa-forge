# vjepa-forge

<p align="center">
  <strong>Forge image and video downstream models from V-JEPA 2.1.</strong>
</p>

<p align="center">
  Train, validate, run inference, benchmark, and export image and video models from one CLI.
</p>

<p align="center">
  <a href="#install">Install</a> |
  <a href="#supported-tasks">Supported Tasks</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#training">Training</a> |
  <a href="#validation">Validation</a> |
  <a href="#inference">Inference</a> |
  <a href="#export">Export</a> |
  <a href="#datasets">Datasets</a> |
  <a href="#configs">Configs</a>
</p>

## Install

```bash
git clone git@github.com:dariussingh/vjepa-forge.git
cd vjepa-forge
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[detection,onnx]
```

## Supported Tasks

| Task | Input | Support |
| --- | --- | --- |
| Classification | Image, Video | Train, val, predict, export |
| Detection | Image, Video | Train, val, predict |
| Segmentation | Image | Train, val, predict |
| Anomaly Detection | Video | Train, val, predict |

## Getting Started

If you already keep datasets and checkpoints outside this repo:

```bash
ln -s ../data data
ln -s ../weights weights
```

Smoke benchmark:

```bash
vjepa-forge benchmark \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml \
  data.image_size=64
```

## Training

```bash
vjepa-forge train \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml
```

Use the same command pattern with any other config file under `vjepa_forge/configs/`.

## Validation

```bash
vjepa-forge val \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml
```

## Inference

```bash
vjepa-forge predict \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml
```

ONNX Runtime inference:

```bash
vjepa-forge predict \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml \
  inference.backend=onnx \
  export.output_path=classification.onnx
```

## Export

```bash
vjepa-forge export \
  config=vjepa_forge/configs/classification/imagenet1k_vitb.yaml \
  export.output_path=classification.onnx
```

## CLI Overrides

Override any config value directly from the command line:

```bash
vjepa-forge train \
  config=vjepa_forge/configs/classification/kinetics400_vitb.yaml \
  data.image_size=224 \
  data.num_frames=8 \
  train.epochs=10
```

## Datasets

`vjepa-forge` expects datasets under `data/` and checkpoints under `weights/`.

ImageNet VID layout:

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

Helper scripts:

```bash
./scripts/download_imagenet_vid.sh
./scripts/download_coco_det.sh
./scripts/download_ucsd_ped2.sh
./scripts/download_cuhk_avenue.sh
./scripts/download_vjepa_weights.sh
```

## Configs

Available configs:

- `vjepa_forge/configs/classification/imagenet1k_vitb.yaml`
- `vjepa_forge/configs/classification/kinetics400_vitb.yaml`
- `vjepa_forge/configs/detection/coco_rf_detr.yaml`
- `vjepa_forge/configs/detection/imagenet_vid_temporal_detr.yaml`
- `vjepa_forge/configs/segmentation/ade20k_semantic.yaml`
- `vjepa_forge/configs/segmentation/coco_instance.yaml`
- `vjepa_forge/configs/anomaly/ucsd_ped2_predictor.yaml`

## License

This repository is released under the MIT License.

Some copied or adapted upstream components are covered by their original notices. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
