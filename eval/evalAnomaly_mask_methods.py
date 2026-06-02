# ============================================================================
# evalAnomaly_mask_methods.py  --  Task 8 main entry point
# ----------------------------------------------------------------------------
# Mask-architecture anomaly segmentation evaluation for EoMT.
#
# Three execution modes (single CLI):
#
#   --mode cache    Run EoMT forward once over the dataset and cache the
#                   last-layer (mask_logits, class_logits) tuples to disk.
#                   HEAVY. Use this exactly once per (checkpoint × dataset).
#
#   --mode score    Read cached logits and compute MSP / MaxLogit /
#                   MaxEntropy / RbA at T=1. CHEAP. Writes results_mask.txt.
#
#   --mode sweep    Read cached logits, sweep MSP at a grid of temperatures,
#                   pick the AuPRC-best T per (checkpoint × dataset).
#
#   --mode full     cache → score → sweep in one invocation.
#
# Three checkpoint presets (resolved to your absolute paths):
#
#   --preset cityscapes   eomt_cityscapes.bin   + cityscapes/semantic/eomt_base_640.yaml
#   --preset coco         eomt_coco.bin         + coco/panoptic/eomt_base_640_2x.yaml
#   --preset finetuned    epoch=9-step=3720.ckpt + cityscapes/semantic/eomt_base_640_finetune.yaml
#
# You can also override with --config and --ckpt explicitly for any preset.
# ============================================================================

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# Local modules (this file lives next to them in eval/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eomt_adapter import build_eomt_from_config, load_eomt_checkpoint, forward_eomt_last_layer
from logits_cache import CacheEntry, cache_path, save_entry, load_entry, iter_cached
from mask_posthoc import METHODS as POSTHOC_METHODS, msp_score
from gt_utils import (
    DEFAULT_EVAL_HW,
    get_gt_path,
    load_gt_mask_for_anomaly,
    dataset_name_from_input,
)
from temperature_sweep import sweep_temperatures, best_temperature_by_auprc

try:
    from ood_metrics import fpr_at_95_tpr
except Exception:                                                   # pragma: no cover
    fpr_at_95_tpr = None                                            # type: ignore[assignment]


# ----------------------------------------------------------------------------
# Reproducibility (matches Task 7 settings — disable cuDNN benchmark)
# ----------------------------------------------------------------------------
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_FILE = Path(__file__).resolve().parent / "results_mask.txt"

DEFAULT_CACHE_ROOT = _REPO_ROOT / "eval" / "_logits_cache"

# Maps preset → (config_relpath, ckpt_abspath)
PRESETS: Dict[str, Tuple[str, str]] = {
    "cityscapes": (
        "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        r"D:\Project\data\eomt_cityscapes.bin",
    ),
    "coco": (
        "eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml",
        r"D:\Project\data\eomt_coco.bin",
    ),
    "finetuned": (
        # NB: the epoch=9-step=3720 checkpoint is a LoRA fine-tune from
        # eomt_coco.bin, trained at img_size=512 with patch_size=16
        # (verified from the checkpoint's hyper_parameters).
        "eomt/configs/dinov2/cityscapes/semantic/eomt_base_512_lora_from_coco_8gb.yaml",
        r"D:\Project\data\checkpoints\epoch=9-step=3720.ckpt",
    ),
}

# Default temperature sweep grid (assignment asks for 0.5 / 0.75 / 1.1 + best).
# We add a few more so "best T" actually has room to move.
DEFAULT_T_GRID: List[float] = [0.5, 0.75, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0]


# ----------------------------------------------------------------------------
# FPR95 helper (uses ood_metrics if available, else manual)
# ----------------------------------------------------------------------------
def _fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    if fpr_at_95_tpr is not None:
        return float(fpr_at_95_tpr(scores, labels))
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresh = np.quantile(pos, 0.05)
    return float((neg >= thresh).mean())


# ----------------------------------------------------------------------------
# Image preprocessing for the EoMT forward pass
# ----------------------------------------------------------------------------
def make_input_transform(model_img_size: Tuple[int, int]):
    """
    Resize image to EoMT's training resolution (square), to ToTensor → [0,1].
    EoMT does its own pixel mean/std normalization inside forward(); do NOT
    normalize here.
    """
    return Compose([
        Resize(model_img_size, Image.BILINEAR),
        ToTensor(),                                  # → [3, H, W]  float32 in [0,1]
    ])


# ============================================================================
# MODE: cache
# ============================================================================
def run_cache_mode(args, model, meta, ckpt_tag: str, config_path: str) -> None:
    """
    Run EoMT once over every image in --input and persist last-layer logits.
    """
    dataset_name = dataset_name_from_input(args.input)
    print(f"\n[cache] dataset = {dataset_name} | ckpt_tag = {ckpt_tag}")
    print(f"[cache] model img_size = {meta['img_size']} | num_q = {meta['num_q']} | "
          f"num_classes = {meta['num_classes']}")

    image_paths: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    image_paths.sort()
    if not image_paths:
        raise SystemExit(f"[cache] no images found at: {args.input}")
    print(f"[cache] {len(image_paths)} images")

    input_tf = make_input_transform(tuple(meta["img_size"]))
    eval_hw = tuple(args.eval_size) if args.eval_size else DEFAULT_EVAL_HW

    for i, p in enumerate(image_paths):
        basename = Path(p).stem
        out_path = cache_path(args.cache_root, ckpt_tag, dataset_name, basename)

        if out_path.exists() and not args.force_cache:
            if i % 50 == 0:
                print(f"  [skip-cached] {basename}")
            continue

        try:
            img = Image.open(p).convert("RGB")
            img_t = input_tf(img)                                # [3, H_in, W_in] in [0,1]
            mask_logits, class_logits = forward_eomt_last_layer(
                model, img_t, device=args.device
            )
            # mask_logits  : [Q, h, w]   on CPU
            # class_logits : [Q, C+1]    on CPU
            entry = CacheEntry(
                mask_logits=mask_logits,
                class_logits=class_logits,
                img_size_hw=eval_hw,
                patch_grid=(mask_logits.shape[-2], mask_logits.shape[-1]),
                gt_path=get_gt_path(p),
                ckpt_tag=ckpt_tag,
                config=str(config_path),
            )
            save_entry(out_path, entry)
            if i % 20 == 0:
                print(f"  [{i+1}/{len(image_paths)}] cached {basename} "
                      f"(patch grid {entry.patch_grid})")
        except Exception as e:
            print(f"  [ERR] {basename}: {e}")
        finally:
            torch.cuda.empty_cache()


# ============================================================================
# MODE: score
# ============================================================================
def _eval_one_method_on_dataset(
    cache_root: str | Path,
    ckpt_tag: str,
    dataset_name: str,
    method_name: str,
    device: str,
    temperature: float = 1.0,
) -> Tuple[float, float, int]:
    score_fn = POSTHOC_METHODS[method_name]
    score_pool, label_pool = [], []
    n_used = 0

    for cpath, basename in iter_cached(cache_root, ckpt_tag, dataset_name):
        entry = load_entry(cpath)
        gt = entry.gt_path
        if not Path(gt).exists():
            continue
        ood_gts = load_gt_mask_for_anomaly(gt, entry.img_size_hw)
        if 1 not in np.unique(ood_gts):
            continue

        # Each method takes raw logits and the target size; T defaults to 1.
        kwargs = dict(target_hw=entry.img_size_hw, device=device)
        if method_name != "maxlogit":
            kwargs["temperature"] = temperature
        else:
            # MaxLogit ignores T by design (no softmax in the path that matters)
            kwargs["temperature"] = temperature
        anomaly_map = score_fn(
            entry.mask_logits, entry.class_logits, **kwargs
        )

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)
        ood_scores = anomaly_map[ood_mask]
        ind_scores = anomaly_map[ind_mask]
        score_pool.append(np.concatenate([ind_scores, ood_scores]))
        label_pool.append(
            np.concatenate([np.zeros_like(ind_scores, dtype=np.uint8),
                            np.ones_like(ood_scores, dtype=np.uint8)])
        )
        n_used += 1

    if n_used == 0:
        return float("nan"), float("nan"), 0
    scores = np.concatenate(score_pool)
    labels = np.concatenate(label_pool)
    return (
        float(average_precision_score(labels, scores) * 100.0),
        float(_fpr95(scores, labels) * 100.0),
        n_used,
    )


def run_score_mode(args, ckpt_tag: str) -> None:
    """
    For each cached dataset under {cache_root}/{ckpt_tag}/, evaluate
    all four mask-based methods at T=1 and write rows to results_mask.txt.
    """
    base = Path(args.cache_root) / ckpt_tag
    if not base.exists():
        raise SystemExit(f"[score] no cache found at: {base}\nRun --mode cache first.")

    datasets = [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    methods = args.methods if args.methods else list(POSTHOC_METHODS.keys())

    print(f"\n[score] ckpt_tag = {ckpt_tag}")
    print(f"[score] datasets = {datasets}")
    print(f"[score] methods  = {methods}")

    rows = []
    for ds in datasets:
        for m in methods:
            auprc, fpr95, n = _eval_one_method_on_dataset(
                args.cache_root, ckpt_tag, ds, m, args.device, temperature=1.0
            )
            row = f"{ckpt_tag:18s} | {m.upper():10s} | {ds:20s} | " \
                  f"AUPRC: {auprc:6.2f}% | FPR@95: {fpr95:6.2f}% | n={n}"
            print(row)
            rows.append(row)

    with open(_RESULTS_FILE, "a") as f:
        f.write("\n# ==== score (T=1) ====\n")
        for r in rows:
            f.write(r + "\n")
    print(f"[score] appended {len(rows)} rows to {_RESULTS_FILE}")


# ============================================================================
# MODE: sweep
# ============================================================================
def run_sweep_mode(args, ckpt_tag: str) -> None:
    """
    Sweep MSP across DEFAULT_T_GRID for every cached dataset under ckpt_tag.
    Logs the full table plus the best-T row per dataset.
    """
    base = Path(args.cache_root) / ckpt_tag
    if not base.exists():
        raise SystemExit(f"[sweep] no cache found at: {base}")

    datasets = [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    T_grid = args.temperatures if args.temperatures else DEFAULT_T_GRID

    print(f"\n[sweep] ckpt_tag = {ckpt_tag}")
    print(f"[sweep] T grid   = {T_grid}")

    rows = []
    for ds in datasets:
        table = sweep_temperatures(
            args.cache_root, ckpt_tag, ds, T_grid, device=args.device
        )
        best_T = best_temperature_by_auprc(table)
        for T, v in table.items():
            tag = "*" if T == best_T else " "
            row = f"{ckpt_tag:18s} | MSP(T={T:.3f}){tag} | {ds:20s} | " \
                  f"AUPRC: {v['auprc']:6.2f}% | FPR@95: {v['fpr95']:6.2f}% | n={v['n_used']}"
            print(row)
            rows.append(row)
        rows.append(f"{ckpt_tag:18s} | best T = {best_T:.3f} | {ds}")
        print(f"  → best T for {ds}: {best_T}")

    with open(_RESULTS_FILE, "a") as f:
        f.write("\n# ==== sweep (MSP at multiple T) ====\n")
        for r in rows:
            f.write(r + "\n")
    print(f"[sweep] appended {len(rows)} rows to {_RESULTS_FILE}")


# ============================================================================
# CLI plumbing
# ============================================================================
def resolve_preset(args) -> Tuple[str, str, str]:
    """
    Resolve --preset / --config / --ckpt to absolute paths and a ckpt_tag.
    """
    if args.config and args.ckpt:
        cfg_path = args.config
        ckpt_path = args.ckpt
        tag = args.tag or Path(args.ckpt).stem
    else:
        if args.preset not in PRESETS:
            raise SystemExit(f"Unknown preset: {args.preset}. "
                             f"Choices: {list(PRESETS.keys())}")
        cfg_rel, ckpt_abs = PRESETS[args.preset]
        cfg_path = str(_REPO_ROOT / cfg_rel)
        ckpt_path = ckpt_abs
        tag = args.tag or args.preset

    if not Path(cfg_path).exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    if not Path(ckpt_path).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    return cfg_path, ckpt_path, tag


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EoMT mask-architecture anomaly evaluation (Task 8)."
    )
    p.add_argument("--mode", required=True, choices=["cache", "score", "sweep", "full"])
    p.add_argument("--preset", choices=list(PRESETS.keys()),
                   help="Quick preset for the 3 expected checkpoints.")
    p.add_argument("--config", help="Override: full path to YAML config.")
    p.add_argument("--ckpt",   help="Override: full path to checkpoint (.bin or .ckpt).")
    p.add_argument("--tag",    help="Override: cache tag for this checkpoint "
                                    "(default: preset name or ckpt stem).")
    p.add_argument("--input",  help="Path to dataset images folder "
                                    "(required for --mode cache and --mode full).")
    p.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT),
                   help="Where to read/write cached logits.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--methods", nargs="+",
                   choices=list(POSTHOC_METHODS.keys()),
                   help="Restrict scoring to a subset of methods.")
    p.add_argument("--temperatures", nargs="+", type=float,
                   help="Temperature grid (sweep mode). Default: "
                        f"{DEFAULT_T_GRID}.")
    p.add_argument("--eval-size", nargs=2, type=int, metavar=("H", "W"),
                   help=f"Spatial size for anomaly evaluation. "
                        f"Default: {DEFAULT_EVAL_HW}.")
    p.add_argument("--masked-attn-enabled", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="EoMT iterative masked-attention. Default off (matches the "
                        "shared_eval / RbA inference protocol).")
    p.add_argument("--force-cache", action="store_true",
                   help="Re-cache even if .pt files already exist.")
    return p


def main():
    args = build_argparser().parse_args()

    cfg_path, ckpt_path, ckpt_tag = resolve_preset(args)
    needs_model = args.mode in ("cache", "full")
    if needs_model and not args.input:
        raise SystemExit("--input is required for --mode cache / --mode full.")

    model, meta = None, None
    if needs_model:
        print(f"[setup] config: {cfg_path}")
        print(f"[setup] ckpt  : {ckpt_path}")
        model, meta = build_eomt_from_config(
            cfg_path,
            masked_attn_enabled=args.masked_attn_enabled,
            device=args.device,
        )
        load_eomt_checkpoint(model, ckpt_path)
        model.eval()

    if args.mode == "cache":
        run_cache_mode(args, model, meta, ckpt_tag, cfg_path)
    elif args.mode == "score":
        run_score_mode(args, ckpt_tag)
    elif args.mode == "sweep":
        run_sweep_mode(args, ckpt_tag)
    elif args.mode == "full":
        run_cache_mode(args, model, meta, ckpt_tag, cfg_path)
        run_score_mode(args, ckpt_tag)
        run_sweep_mode(args, ckpt_tag)


if __name__ == "__main__":
    main()
