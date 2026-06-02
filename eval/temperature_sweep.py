# ============================================================================
# temperature_sweep.py
# ----------------------------------------------------------------------------
# Offline temperature search over cached EoMT logits.
#
# Inputs:  cache_root, ckpt_tag, dataset_name (which cache to read).
# Output:  table {T -> {AuPRC, FPR95}} + the AuPRC-best T.
#
# Reuses mask_posthoc.msp_score with the temperature kwarg, so we never
# re-run the model. The cost of one temperature is one renormalized softmax
# + one einsum + one max — milliseconds per image on GPU.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

try:
    from ood_metrics import fpr_at_95_tpr
except Exception:                                                   # pragma: no cover
    fpr_at_95_tpr = None                                            # type: ignore[assignment]

from logits_cache import iter_cached, load_entry
from mask_posthoc import msp_score
from gt_utils import load_gt_mask_for_anomaly


def _fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Local fallback if ood_metrics isn't installed (it ships with the
    Task-7 env so this should be a no-op).
    """
    if fpr_at_95_tpr is not None:
        return float(fpr_at_95_tpr(scores, labels))
    # Manual FPR@95: find threshold giving TPR=0.95, then FPR at that threshold.
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresh = np.quantile(pos, 0.05)            # 95% of anomalies above this
    return float((neg >= thresh).mean())


def evaluate_msp_at_T(
    cache_root: str | Path,
    ckpt_tag: str,
    dataset_name: str,
    temperature: float,
    device: str = "cuda",
) -> Tuple[float, float, int]:
    """
    Compute AuPRC and FPR@95 for MSP at one temperature on one dataset.

    Returns (auprc_pct, fpr95_pct, n_images_used).
    """
    score_pool: List[np.ndarray] = []
    label_pool: List[np.ndarray] = []
    n_used = 0

    for cache_path, basename in iter_cached(cache_root, ckpt_tag, dataset_name):
        entry = load_entry(cache_path)
        gt_path = entry.gt_path
        if not Path(gt_path).exists():
            continue
        ood_gts = load_gt_mask_for_anomaly(gt_path, entry.img_size_hw)
        # Skip frames with no anomaly pixels — matches Task-7 protocol.
        if 1 not in np.unique(ood_gts):
            continue

        anomaly_map = msp_score(
            entry.mask_logits, entry.class_logits, entry.img_size_hw,
            temperature=temperature, device=device,
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
    auprc = average_precision_score(labels, scores) * 100.0
    fpr95 = _fpr95(scores, labels) * 100.0
    return float(auprc), float(fpr95), n_used


def sweep_temperatures(
    cache_root: str | Path,
    ckpt_tag: str,
    dataset_name: str,
    temperatures: Sequence[float],
    device: str = "cuda",
) -> Dict[float, Dict[str, float]]:
    """
    Run MSP at each temperature on a single dataset; return the result table.
    """
    table: Dict[float, Dict[str, float]] = {}
    for T in temperatures:
        auprc, fpr95, n = evaluate_msp_at_T(
            cache_root, ckpt_tag, dataset_name, temperature=T, device=device
        )
        table[float(T)] = {"auprc": auprc, "fpr95": fpr95, "n_used": n}
    return table


def best_temperature_by_auprc(table: Dict[float, Dict[str, float]]) -> float:
    """Argmax over AuPRC; ties broken by lower T (sharper distribution wins)."""
    valid = [(T, v["auprc"]) for T, v in table.items() if not np.isnan(v["auprc"])]
    if not valid:
        return float("nan")
    valid.sort(key=lambda x: (-x[1], x[0]))
    return valid[0][0]
