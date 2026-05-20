# ============================================================================
# gt_utils.py
# ----------------------------------------------------------------------------
# Ground-truth and path utilities shared by the mask-based anomaly pipeline.
# These mirror the conventions established in evalAnomaly_methods.py (Task 7)
# so results between ERFNet and EoMT are directly comparable.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize


# Standard anomaly-evaluation resolution. Matches Task 7 (evalAnomaly_methods.py).
DEFAULT_EVAL_HW = (512, 1024)


def make_target_transform(target_hw: Tuple[int, int] = DEFAULT_EVAL_HW):
    """Nearest-neighbour resize so label IDs are preserved exactly."""
    return Compose([Resize(target_hw, Image.NEAREST)])


def get_gt_path(image_path: str) -> str:
    """
    Image → GT mask path. Identical rules to evalAnomaly_methods.py:
      - swap 'images/' for 'labels_masks/'
      - extension fix-ups for the three formats that use a different
        extension between image and mask
    """
    gt = image_path.replace("images", "labels_masks")
    if "RoadObsticle21" in gt:
        gt = gt.replace(".webp", ".png")
    if "fs_static" in gt:
        gt = gt.replace(".jpg", ".png")
    if "RoadAnomaly" in gt and "RoadAnomaly21" not in gt:
        gt = gt.replace(".jpg", ".png")
    return gt


def load_gt_mask_for_anomaly(
    gt_path: str,
    target_hw: Tuple[int, int] = DEFAULT_EVAL_HW,
) -> np.ndarray:
    """
    Load and remap an anomaly GT mask to the standard {0 normal, 1 anomaly,
    255 ignore} convention, resized to target_hw with nearest-neighbour.

    Dataset rules:
      RoadAnomaly (legacy split) : raw labels are 0/2 → remap 2→1
      RoadAnomaly21              : already 0/1/255 — leave as-is
      RoadObsticle21             : already 0/1/255 — leave as-is
      FS LostFound (full)        : already 0/1/255 — leave as-is
      fs_static                  : already 0/1/255 — leave as-is
    """
    transform = make_target_transform(target_hw)
    mask = Image.open(gt_path)
    mask = transform(mask)
    arr = np.array(mask)

    # Only the legacy 'RoadAnomaly' split needs remapping
    if "RoadAnomaly" in gt_path and "RoadAnomaly21" not in gt_path:
        arr = np.where(arr == 2, 1, arr)
    return arr


def dataset_name_from_input(input_dir: str) -> str:
    """
    Extract a stable dataset name from a path like
      .../Anomaly_Validation_Datasets/RoadAnomaly21/images
    → 'RoadAnomaly21'
    """
    parent = Path(input_dir.rstrip("/\\")).parent
    return parent.name
