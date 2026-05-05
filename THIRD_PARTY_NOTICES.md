# Third-Party Notices

`vjepa-forge` is distributed under the MIT license in [`LICENSE`](./LICENSE).

This repository includes code derived from third-party open-source projects. Original copyright and license notices are retained in the relevant source files.

## Upstream Repositories

- [facebookresearch/vjepa](https://github.com/facebookresearch/vjepa) — original V-JEPA architecture, encoder and predictor design
- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) — V-JEPA 2 / 2.1 backbone weights, ViT patch embedding, tubelet tokenization
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection head architecture, DFL loss, task-aligned assigner logic (`vjepa_forge/heads/detection/`, `vjepa_forge/losses/detection/`)
- [roboflow/rf-detr](https://github.com/roboflow/rf-detr) — Hungarian matcher, SetCriterion, box operations (`vjepa_forge/heads/detection/`)
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) — training methodology reference for frozen-backbone fine-tuning

## Notes

- Upstream license terms continue to apply to the relevant third-party source files included in this repository.
- See the headers in the relevant files for the original notices.
- Reimplemented utilities (detection heads, losses, matchers) are independent implementations inspired by the upstream work; they do not vendor upstream source code.
