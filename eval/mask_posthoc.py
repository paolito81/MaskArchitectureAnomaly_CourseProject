# Anomaly-score functions for EoMT mask-classification outputs.

# All five methods operate on a SINGLE image's raw last-layer outputs:
#     mask_logits  : [Q, h, w]   (pre-sigmoid; h,w = patch grid after ScaleBlocks)
#     class_logits : [Q, C+1]    (pre-softmax; last slot = "no-object")
# plus a target spatial size (H, W) that the caller wants the anomaly map at.

# Returns a numpy array [H, W] where HIGHER = MORE ANOMALOUS for every method




from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F



def pp_class_scores(
    mask_logits: torch.Tensor,   # [Q, h, w]  raw
    class_logits: torch.Tensor,  # [Q, C+1]   raw
    target_hw: Tuple[int, int],  # (H, W) the output anomaly map size
    temperature: float = 1.0,
    drop_no_object: bool = True,
    device: str = "cuda",
) -> torch.Tensor:

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



def _renormalize_per_pixel(pp_class: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Force per-pixel sum-to-one across the C class channels.
    Needed for MSP and MaxEntropy because the einsum drops the no-object slot.
    """
    s = pp_class.sum(dim=0, keepdim=True).clamp_min(eps)    # [1, H, W]
    return pp_class / s                                     # [C, H, W]


# Method 1: MSP
def msp_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:

    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    p = _renormalize_per_pixel(pp)
    max_p = p.max(dim=0).values                                     
    return (1.0 - max_p).detach().cpu().numpy()


# Method 2: MaxLogit
def maxlogit_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:

    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    max_s = pp.max(dim=0).values                                    
    return (-max_s).detach().cpu().numpy()


# Method 3: MaxEntropy
def maxentropy_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    normalize: bool = True,
    device: str = "cuda",
) -> np.ndarray:

    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    p = _renormalize_per_pixel(pp)
    log_p = torch.log(p + 1e-10)
    H = -(p * log_p).sum(dim=0)                                     
    if normalize:
        C = pp.shape[0]
        H = H / math.log(C)         # bounded [0, 1]
    return H.detach().cpu().numpy()


# Method 4: RbA - Rejected by All
def rba_score(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    target_hw: Tuple[int, int],
    temperature: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:

    pp = pp_class_scores(mask_logits, class_logits, target_hw, temperature, device=device)
    known_mass = pp.sum(dim=0)                                       
    return (-known_mass).detach().cpu().numpy()



METHODS = {
    "msp":        msp_score,
    "maxlogit":   maxlogit_score,
    "maxentropy": maxentropy_score,
    "rba":        rba_score,
}
