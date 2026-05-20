# ============================================================================
# mask_posthoc.py
# ----------------------------------------------------------------------------
# Anomaly-score functions for EoMT mask-classification outputs.
#
# All five methods operate on a SINGLE image's raw last-layer outputs:
#     mask_logits  : [Q, h, w]   (pre-sigmoid; h,w = patch grid after ScaleBlocks)
#     class_logits : [Q, C+1]    (pre-softmax; last slot = "no-object")
# plus a target spatial size (H, W) that the caller wants the anomaly map at.
#
# Returns a numpy array [H, W] where HIGHER = MORE ANOMALOUS for every method
# (sign convention matches Task-7 / ERFNet baselines).
#
# Reduction (canonical):
#     S_c(p) = sum_q  sigmoid(mask_logit_q,p) * softmax(class_logit_q / T)_c
#   with c in {0..C-1}  (the no-object slot, c=C, is dropped)
#
# The per-pixel-per-known-class score S has C channels, values in [0,1], and
# does NOT sum to 1 across c (because the no-object mass is dropped). For MSP
# and MaxEntropy we explicitly re-normalize per pixel to form a real
# probability distribution; for MaxLogit and RbA we use S as-is.
# ============================================================================

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Core reduction
# ----------------------------------------------------------------------------
def pp_class_scores(
    mask_logits: torch.Tensor,   # [Q, h, w]  raw
    class_logits: torch.Tensor,  # [Q, C+1]   raw
    target_hw: Tuple[int, int],  # (H, W) the output anomaly map size
    temperature: float = 1.0,
    drop_no_object: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the per-pixel-per-known-class score tensor S of shape [C, H, W].

    Steps:
      1. mask_prob_q(p)  = sigmoid(mask_logits)               at patch resolution
      2. bilinear upsample mask probabilities to (H, W)
      3. class_prob_q(c) = softmax(class_logits / T) , drop no-object slot
      4. einsum to marginalize over queries → S_c(p)

    Note we upsample the MASK PROB (not raw mask_logits) because sigmoid is
    monotonic and the values are bounded — bilinear stays in [0,1].

    target_hw is the resolution at which the anomaly metric will be computed,
    typically the resolution to which the GT mask was resized (e.g. 512x1024).
    """
    mask_logits = mask_logits.to(device).float()
    class_logits = class_logits.to(device).float()

    # [Q, h, w] -> [Q, H, W]  via sigmoid then bilinear interp
    mask_prob = torch.sigmoid(mask_logits)                # [Q, h, w]
    mask_prob = mask_prob.unsqueeze(0)                    # [1, Q, h, w]
    mask_prob = F.interpolate(
        mask_prob, size=target_hw, mode="bilinear", align_corners=False
    )[0]                                                  # [Q, H, W]

    # [Q, C+1] -> [Q, C]  via tempered softmax, drop the no-object slot
    cls_prob_full = F.softmax(class_logits / temperature, dim=-1)  # [Q, C+1]
    if drop_no_object:
        cls_prob = cls_prob_full[..., :-1]                          # [Q, C]
    else:
        cls_prob = cls_prob_full                                    # [Q, C+1]

    # Marginalize over queries
    #   S[c, h, w] = sum_q  mask_prob[q, h, w] * cls_prob[q, c]
    pp_class = torch.einsum("qhw,qc->chw", mask_prob, cls_prob)    # [C, H, W]
    return pp_class                                                 # values in [0,1]


# ----------------------------------------------------------------------------
# Helper: renormalize S(p) into a real probability distribution per pixel.
# ----------------------------------------------------------------------------
def _renormalize_per_pixel(pp_class: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Force per-pixel sum-to-one across the C class channels.
    Needed for MSP and MaxEntropy because the einsum drops the no-object slot.
    """
    s = pp_class.sum(dim=0, keepdim=True).clamp_min(eps)   # [1, H, W]
    return pp_class / s                                     # [C, H, W]


# ----------------------------------------------------------------------------
# Method 1: MSP
# ----------------------------------------------------------------------------
def msp_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    MSP for mask architectures.

        score(p) = 1 - max_c P(c|p)
        P(c|p)   = S_c(p) / sum_c' S_c'(p)

    Higher score = lower max confidence = more anomalous.
    """
    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    p = _renormalize_per_pixel(pp)
    max_p = p.max(dim=0).values                                     # [H, W]
    return (1.0 - max_p).detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Method 2: MaxLogit
# ----------------------------------------------------------------------------
def maxlogit_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    MaxLogit adapted to mask architectures.

        score(p) = - max_c S_c(p)

    We do NOT renormalize. The raw magnitude carries information: if
    no class has any meaningful mask×class mass at p, the max is small
    (close to 0), so the score is large (close to 0 with negative sign
    flipped). Negation keeps the convention `higher = more anomalous`.
    """
    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    max_s = pp.max(dim=0).values                                    # [H, W]
    return (-max_s).detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Method 3: MaxEntropy
# ----------------------------------------------------------------------------
def maxentropy_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    normalize: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    Shannon entropy of the renormalized per-pixel-per-class distribution.

        H(p) = - sum_c P(c|p) * log P(c|p)

    Optionally normalize by log(C) to map entropy into [0, 1].
    """
    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    p = _renormalize_per_pixel(pp)
    log_p = torch.log(p + 1e-10)
    H = -(p * log_p).sum(dim=0)                                     # [H, W]
    if normalize:
        C = pp.shape[0]
        H = H / math.log(C)                                         # bounded [0, 1]
    return H.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Method 4: RbA — Rejected by All
# ----------------------------------------------------------------------------
def rba_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """
    RbA (Nayal et al., ICCV 2023): negative sum of per-pixel-per-known-class
    score across all C known classes (no-object slot dropped before the sum,
    which is precisely the mass that should be 'rejected').

        rba(p) = - sum_c S_c(p)
               = - sum_q sigmoid(mask_logit_q,p) * (1 - p_{q,no_object})

    Intuition (rewritten via the second equality): for each pixel, we sum
    across queries the product of (1) "this query owns this pixel" and
    (2) "this query represents a known class". If every query either does
    not own the pixel OR represents no-object at it, the sum collapses
    near zero → the pixel is rejected by all → high anomaly.

    Sign convention: higher = more anomalous.
    """
    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    known_mass = pp.sum(dim=0)                                       # [H, W]
    return (-known_mass).detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Method registry — convenience for the orchestrator
# ----------------------------------------------------------------------------
METHODS = {
    "msp":        msp_score,
    "maxlogit":   maxlogit_score,
    "maxentropy": maxentropy_score,
    "rba":        rba_score,
}
