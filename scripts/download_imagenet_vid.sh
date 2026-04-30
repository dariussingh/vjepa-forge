#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
ImageNet VID cannot be redistributed by this repo.

Expected layout:
  data/imagenet_vid/
    Data/VID/train/...
    Data/VID/val/...
    Annotations/VID/train/...
    Annotations/VID/val/...

Example recipe:
  vjepa-forge train config=vjepa_forge/configs/detection/imagenet_vid_temporal_detr.yaml

Place the extracted ImageNet VID dataset under ./data/imagenet_vid and rerun.
EOF
