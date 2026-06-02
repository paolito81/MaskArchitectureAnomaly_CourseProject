# ============================================================================
# eomt_adapter.py
# ----------------------------------------------------------------------------
# Adapter layer between Task-7's pixel-based eval scaffolding and the EoMT
# (mask-classification) model in eomt/models/eomt.py.
#
# Responsibilities:
#   1. Build an EoMT model from a YAML config (using the same construction
#      logic as eomt/eval_shared_miou.py, so we never diverge).
#   2. Unified checkpoint loader for .bin (raw state_dict) and .ckpt
#      (PyTorch Lightning) files.
#   3. Single-shot inference helper that takes an image tensor at the model's
#      training resolution and returns the LAST-LAYER (mask_logits, class_logits)
#      tuple, ready for downstream mask-architecture anomaly scoring.
#
# Tensor shapes throughout:
#   image_in        : [1, 3, H_in, W_in]            (H_in == W_in == config img_size)
#   mask_logits     : [1, Q, h, w]                  RAW, pre-sigmoid; (h,w) = patch_grid * scaleblocks
#   class_logits    : [1, Q, C+1]                   RAW, pre-softmax; last slot is "no-object"
# ============================================================================

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import yaml

# ----------------------------------------------------------------------------
# Path bootstrap: this file lives in `eval/` but EoMT modules live in `eomt/`.
# We add the eomt directory to sys.path so `from datasets.*`, `from models.*`,
# `from training.*` work regardless of which cwd the user runs us from.
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_EOMT_DIR = _REPO_ROOT / "eomt"
if str(_EOMT_DIR) not in sys.path:
    sys.path.insert(0, str(_EOMT_DIR))


# ----------------------------------------------------------------------------
# Reflection helpers — same approach as eval_shared_miou.py
# ----------------------------------------------------------------------------
def _import_class(class_path: str):
    """Resolve a dotted class path like 'models.eomt.EoMT' to the class object."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _get_constructor_default(cls, param_name):
    sig = inspect.signature(cls.__init__)
    param = sig.parameters[param_name]
    if param.default is inspect._empty:
        raise ValueError(f"{cls.__name__}.__init__ has no default for {param_name}")
    return param.default


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_eomt_from_config(
    config_path: str,
    masked_attn_enabled: bool = False,
    device: str = "cuda",
) -> Tuple[nn.Module, dict]:
    """
    Build an EoMT Lightning module from a YAML config.

    masked_attn_enabled is forced False at inference (matches notebook eval
    and the original RbA / EoMT-as-decoder evaluation protocol — no iterative
    mask refinement during single-shot inference).

    Returns:
        model           : nn.Module on `device`, in eval() mode.
        meta            : dict with {num_classes, img_size, num_q}
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- data class → discover num_classes and img_size ---
    src_data_cls = _import_class(config["data"]["class_path"])
    src_data_kwargs = config["data"].get("init_args", {}) or {}
    num_classes = src_data_kwargs.get(
        "num_classes", _get_constructor_default(src_data_cls, "num_classes")
    )
    img_size = tuple(
        src_data_kwargs.get(
            "img_size", _get_constructor_default(src_data_cls, "img_size")
        )
    )

    # --- encoder ---
    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_cls = _import_class(enc_cfg["class_path"])
    encoder = enc_cls(img_size=img_size, **enc_cfg.get("init_args", {}))

    # --- network (EoMT itself) ---
    net_cfg = config["model"]["init_args"]["network"]
    net_cls = _import_class(net_cfg["class_path"])
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = net_cls(
        encoder=encoder,
        num_classes=num_classes,
        masked_attn_enabled=masked_attn_enabled,
        **net_kwargs,
    )
    num_q = network.num_q

    # --- Lightning model wrapper ---
    model_cfg = config["model"]
    model_cls = _import_class(model_cfg["class_path"])
    model_kwargs = {k: v for k, v in model_cfg["init_args"].items() if k != "network"}
    if "stuff_classes" in src_data_kwargs:
        model_kwargs["stuff_classes"] = src_data_kwargs["stuff_classes"]

    model = model_cls(
        network=network,
        img_size=img_size,
        num_classes=num_classes,
        **model_kwargs,
    ).eval()

    model = model.to(device)
    meta = dict(num_classes=num_classes, img_size=img_size, num_q=num_q)
    return model, meta


# ----------------------------------------------------------------------------
# Unified checkpoint loader (.bin = raw state_dict | .ckpt = Lightning dict)
# ----------------------------------------------------------------------------
def load_eomt_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """
    Loads either:
      - a .bin file: torch.save(state_dict) — raw dict of tensors, keys like
        'network.encoder.backbone....' or just 'encoder.backbone....'
      - a .ckpt file: PyTorch Lightning checkpoint with top-level keys
        like {'state_dict': ..., 'optimizer_states': ..., ...}

    We unwrap the Lightning wrapper if present, drop loss/criterion buffers
    that don't belong to the inference graph, and load with strict=False so
    a single missing buffer like attn_mask_probs does not abort.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    # Filter out training-only entries that aren't part of the inference graph.
    # 'criterion.empty_weight' is a buffer on MaskClassificationLoss — not on the model proper.
    state_dict = {k: v for k, v in state_dict.items() if "criterion." not in k}

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys:
        print(f"[ckpt] {len(incompatible.missing_keys)} missing keys "
              f"(usually attn_mask_probs buffer — safe)")
    if incompatible.unexpected_keys:
        print(f"[ckpt] {len(incompatible.unexpected_keys)} unexpected keys "
              f"(usually training-only state — safe)")


# ----------------------------------------------------------------------------
# Single-shot inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def forward_eomt_last_layer(
    model: nn.Module,
    image_chw: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single image through EoMT at the model's native img_size and
    return ONLY the last-layer mask & class logits.

    Args:
        model      : an EoMT-wrapping LightningModule (e.g. MaskClassificationSemantic).
                     We call `model.network(...)` directly to bypass any per-layer-list
                     bookkeeping in eval_step.
        image_chw  : Tensor [3, H_in, W_in] OR [1, 3, H_in, W_in] in [0,1] float — the
                     EoMT model handles its own (x - mean) / std internally, so DO NOT
                     pre-normalize here.
        device     : where to run.

    Returns:
        mask_logits  : [Q, h, w]   fp32, raw (pre-sigmoid)
        class_logits : [Q, C+1]    fp32, raw (pre-softmax)
        Both on CPU to make caching easy.
    """
    if image_chw.dim() == 3:
        image_chw = image_chw.unsqueeze(0)
    image_chw = image_chw.to(device, non_blocking=True).float()

    mask_logits_per_layer, class_logits_per_layer = model.network(image_chw)
    # Last layer = full-depth output. Earlier layers are intermediate
    # supervision heads from inside the transformer (training only).
    mask_logits = mask_logits_per_layer[-1][0].detach().to("cpu")   # [Q, h, w]
    class_logits = class_logits_per_layer[-1][0].detach().to("cpu") # [Q, C+1]
    return mask_logits, class_logits
