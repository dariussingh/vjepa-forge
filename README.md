# vjepa-forge

Canonical image/video training framework for V-JEPA 2.1-style downstream tasks.

## CLI

```bash
forge classify train model=vjepa21-b.yaml data=kinetics400.yaml
forge detect train model=vjepa21-rfdetr.yaml data=coco.yaml
forge segment train model=vjepa21-vos.yaml data=davis.yaml
forge anomaly train model=vjepa21-predictor.yaml data=ucsd_ped2.yaml
```

Converters are the only place external dataset formats belong:

```bash
forge convert coco source=/data/coco out=/data/coco_forge task=detect media=image
forge convert kinetics source=/data/kinetics out=/data/kinetics_forge task=classify media=video
```

## Dataset Format

Each dataset is described by a `forge.yaml` file with one `task` and one `media`:

```yaml
path: /data/my_dataset
task: detect
media: image
names:
  0: person
splits:
  train: splits/train.txt
  val: splits/val.txt
labels:
  format: forge-yolo
  root: labels
```

Image runs use `x: [B, C, H, W]`. Video runs use `x: [B, T, C, H, W]`.

Video annotations may be frame-aware, but model execution is always clip-level.

## Python API

```python
from vjepa_forge import ForgeModel

model = ForgeModel("vjepa21-rfdetr.yaml", data={"task": "detect", "media": "image", "image_size": 64})
```

## Notes

- `media` is only `image` or `video`
- there is no mixed-media dataset type
- runtime parsing uses one canonical Forge text-label parser
- dataset-specific runtime loaders are replaced by converters plus task/media loaders
