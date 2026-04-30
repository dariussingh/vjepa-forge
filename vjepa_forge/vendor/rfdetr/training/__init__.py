# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightweight training package init.

This local vendored copy keeps ``rfdetr.training`` importable without the full
PyTorch Lightning stack so utility modules such as ``param_groups`` can be used
from the V-JEPA integration path.
"""

__all__: list[str] = []
