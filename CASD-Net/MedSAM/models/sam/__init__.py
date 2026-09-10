# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Slimmed entry point for CASD-Net: only the SAM encoder builders and the
# model registry are exposed. The interactive `SamPredictor` and
# `SamAutomaticMaskGenerator` (which pull in pycocotools / matting helpers)
# are intentionally omitted so that this vendored dependency stays minimal.
from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
