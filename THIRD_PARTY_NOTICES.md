# Third-Party Notices

`vjepa-forge` is distributed under the MIT license in [`LICENSE`](./LICENSE).

This repository also contains source code copied and adapted from upstream projects whose original copyright and license notices have been retained in the relevant files.

## Meta / Facebook Research V-JEPA Source

The following local modules include copied or adapted upstream source from the V-JEPA / V-JEPA 2.x codebases:

- `vjepa_forge/models/vision_transformer.py`
- `vjepa_forge/models/predictor.py`
- `vjepa_forge/models/utils/modules.py`
- `vjepa_forge/models/utils/patch_embed.py`
- `vjepa_forge/utils/checkpoint_loader.py`
- `vjepa_forge/utils/logging.py`
- `vjepa_forge/utils/tensors.py`
- `vjepa_forge/masks/utils.py`
- `vjepa_forge/backbones/factory.py`

Those files retain their upstream copyright headers and licensing notices.

For the V-JEPA-derived files copied from the local source repository, the upstream repository license is MIT.

## Local Adaptation Policy

When upstream code was copied into `vjepa_forge`, it was adapted to:

- remove bridge namespaces such as `src`, `app`, and `vendor`
- use direct local imports under `vjepa_forge`
- keep only the relevant runtime subset needed by this repository

These adaptations do not remove or replace the original copyright and license notices embedded in the copied files.
