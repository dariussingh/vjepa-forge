# vjepa-forge

**vjepa-forge is an open-source framework for building image and video perception models with V-JEPA 2.1 backbones, dense predictive features, DETR/RF-DETR-style detection heads, and reproducible benchmark recipes.**

GitHub description:

```text
Forge image and video downstream models from V-JEPA 2.1: classification, detection, segmentation, and predictor-based anomaly detection with reproducible recipes.
```

## Commands

```bash
vjepa-forge train recipe=vjepa_forge/recipes/classification/imagenet1k_vitb.yaml
vjepa-forge val recipe=vjepa_forge/recipes/detection/coco_rf_detr.yaml
vjepa-forge predict recipe=vjepa_forge/recipes/segmentation/ade20k_semantic.yaml source=example.jpg
vjepa-forge export recipe=vjepa_forge/recipes/anomaly/ucsd_ped2_predictor.yaml format=onnx
```

## Notes

- Images are treated as `T=1` videos throughout the public API.
- The repo vendors the required V-JEPA and RF-DETR runtime code locally.
- Ultralytics and ONNX Runtime are optional extras.
