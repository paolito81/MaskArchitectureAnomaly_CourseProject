# logits_cache.py
# Disk-backed per-image cache for EoMT last-layer outputs.


# Layout:
#   {cache_root}/
#       {ckpt_tag}/                       example: "eomt_cityscapes" / "eomt_coco" / "finetune"
#           {dataset_name}/               example: "RoadAnomaly21"
#               {image_basename}.pt       one file per image


# Each .pt is a small dict. tensors are stored on CPU.
# Mask logits are saved in fp16 (lossy by ~3 decimals; harmless for ranking metrics). 
# Class logits are kept fp32 because they're tiny.



from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class CacheEntry:
    """Schema for one cached image."""
    mask_logits: torch.Tensor      # [Q, h, w]   fp16
    class_logits: torch.Tensor     # [Q, C+1]    fp32
    img_size_hw: Tuple[int, int]   
    patch_grid: Tuple[int, int]  
    gt_path: str                   
    ckpt_tag: str
    config: str


def cache_path(
    cache_root: str | Path,
    ckpt_tag: str,
    dataset_name: str,
    image_basename: str,
) -> Path:
    p = Path(cache_root) / ckpt_tag / dataset_name / f"{image_basename}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_entry(path: Path, entry: CacheEntry) -> None:
    obj = {
        "mask_logits":  entry.mask_logits.detach().to(torch.float16).contiguous(),
        "class_logits": entry.class_logits.detach().to(torch.float32).contiguous(),
        "img_size_hw":  tuple(entry.img_size_hw),
        "patch_grid":   tuple(entry.patch_grid),
        "gt_path":      entry.gt_path,
        "ckpt_tag":     entry.ckpt_tag,
        "config":       entry.config,
    }

    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_entry(path: Path) -> CacheEntry:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return CacheEntry(
        mask_logits  = obj["mask_logits"].to(torch.float32),
        class_logits = obj["class_logits"].to(torch.float32),
        img_size_hw  = tuple(obj["img_size_hw"]),
        patch_grid   = tuple(obj["patch_grid"]),
        gt_path      = obj["gt_path"],
        ckpt_tag     = obj["ckpt_tag"],
        config       = obj["config"],
    )


def iter_cached(cache_root: str | Path, ckpt_tag: str, dataset_name: str):
    d = Path(cache_root) / ckpt_tag / dataset_name
    if not d.exists():
        return
    for f in sorted(d.glob("*.pt")):
        yield f, f.stem
