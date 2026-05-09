# Shared Evaluation Skeleton

The goal is:
- keep the original repo behavior untouched
- add one shared-eval mapping module
- add one shared semantic validation module
- drive it with configs

## 1. Mapping module

File:
`eomt/shared_eval/shared.py`

Skeleton:

```python
import torch

IGNORE_INDEX = 255

SHARED_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
]

SHARED_NAME_TO_ID = {name: i for i, name in enumerate(SHARED_CLASSES)}

CITYSCAPES_TO_SHARED = {
    11: SHARED_NAME_TO_ID["person"],
    13: SHARED_NAME_TO_ID["car"],
    14: SHARED_NAME_TO_ID["truck"],
    15: SHARED_NAME_TO_ID["bus"],
    17: SHARED_NAME_TO_ID["motorcycle"],
    18: SHARED_NAME_TO_ID["bicycle"],
    6: SHARED_NAME_TO_ID["traffic light"],
}

COCO_TO_SHARED = {
    0: SHARED_NAME_TO_ID["person"],
    2: SHARED_NAME_TO_ID["car"],
    7: SHARED_NAME_TO_ID["truck"],
    5: SHARED_NAME_TO_ID["bus"],
    3: SHARED_NAME_TO_ID["motorcycle"],
    1: SHARED_NAME_TO_ID["bicycle"],
    9: SHARED_NAME_TO_ID["traffic light"],
}

def remap_target_ids(target: torch.Tensor, id_map: dict[int, int], ignore_index: int = IGNORE_INDEX):
    remapped = target.new_full(target.shape, ignore_index)
    for src_id, dst_id in id_map.items():
        remapped[target == src_id] = dst_id
    return remapped

def remap_logits(logits: torch.Tensor, id_map: dict[int, int], num_shared_classes: int):
    shared = logits.new_zeros((num_shared_classes, *logits.shape[1:]))
    for src_id, dst_id in id_map.items():
        shared[dst_id] += logits[src_id]
    return shared
```

## 2. Shared validation module

File:
`eomt/training/mask_classification_semantic_shared.py`

Idea:
- inherit from `MaskClassificationSemantic`
- override metric initialization
- override `eval_step`

Skeleton:

```python
from typing import Optional, List
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

from training.mask_classification_semantic import MaskClassificationSemantic
from shared_eval.shared import (
    IGNORE_INDEX,
    SHARED_CLASSES,
    CITYSCAPES_TO_SHARED,
    COCO_TO_SHARED,
    remap_logits,
    remap_target_ids,
)

class MaskClassificationSemanticShared(MaskClassificationSemantic):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = IGNORE_INDEX,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        src_to_shared: str = "cityscapes",
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.src_to_shared_name = src_to_shared
        self.cityscapes_to_shared = CITYSCAPES_TO_SHARED
        self.num_shared_classes = len(SHARED_CLASSES)

        if src_to_shared == "cityscapes":
            self.src_to_shared = CITYSCAPES_TO_SHARED
        elif src_to_shared == "coco":
            self.src_to_shared = COCO_TO_SHARED
        else:
            raise ValueError(f"Unknown src_to_shared: {src_to_shared}")

        num_blocks = self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_shared_classes,
                    validate_args=False,
                    ignore_index=self.ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        shared_targets = [
            remap_target_ids(t, self.cityscapes_to_shared, self.ignore_idx)
            for t in targets
        ]

        for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            shared_logits = [
                remap_logits(x, self.src_to_shared, self.num_shared_classes)
                for x in logits
            ]

            self.update_metrics_semantic(shared_logits, shared_targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0],
                    shared_targets[0],
                    shared_logits[0],
                    log_prefix,
                    i,
                    batch_idx,
                )
```

## 3. Config for Cityscapes-trained checkpoint in shared eval

File:
`eomt/configs/shared_eval/cityscapes_shared.yaml`

Skeleton:

```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.wandb.WandbLogger
    init_args:
      project: "eomt_shared_eval"
      name: "cityscapes_shared_eval"

model:
  class_path: training.mask_classification_semantic_shared.MaskClassificationSemanticShared
  init_args:
    src_to_shared: cityscapes
    attn_mask_annealing_enabled: True
    attn_mask_annealing_start_steps: [3317, 8292, 13268]
    attn_mask_annealing_end_steps: [6634, 11609, 16585]
    network:
      class_path: models.eomt.EoMT
      init_args:
        num_q: 100
        num_blocks: 3
        encoder:
          class_path: models.vit.ViT
          init_args:
            backbone_name: vit_base_patch14_reg4_dinov2

data:
  class_path: datasets.cityscapes_semantic.CityscapesSemantic
```

## 4. Config for COCO-trained checkpoint on Cityscapes shared eval

Important note:
`main.py` links `data.num_classes -> model.num_classes`, so configs alone won’t let a COCO 133-class checkpoint coexist with Cityscapes 19-class data under the current CLI setup.

So for the COCO shared eval, you have two options:

- best long-term: add a dedicated standalone eval script
- quick workaround: temporarily bypass the CLI linking logic for this special case

I strongly recommend the standalone script.

## 5. Standalone evaluator skeleton

File:
`eomt/shared_eval/eval_shared_miou.py`

Skeleton:

```python
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from shared_eval.shared import SHARED_CLASSES, IGNORE_INDEX, remap_logits, remap_target_ids, CITYSCAPES_TO_SHARED, COCO_TO_SHARED

def evaluate(model, loader, src_to_shared):
    metric = MulticlassJaccardIndex(
        num_classes=len(SHARED_CLASSES),
        average=None,
        ignore_index=IGNORE_INDEX,
    ).to(model.device)

    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            # adapt this for your batch format
            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            mask_logits = torch.nn.functional.interpolate(
                mask_logits_per_layer[-1],
                model.img_size,
                mode="bilinear",
            )
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits,
                class_logits_per_layer[-1],
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            per_pixel_targets = model.to_per_pixel_targets_semantic(targets, IGNORE_INDEX)

            for logit, target in zip(logits, per_pixel_targets):
                shared_logits = remap_logits(logit, src_to_shared, len(SHARED_CLASSES))
                shared_target = remap_target_ids(target, CITYSCAPES_TO_SHARED, IGNORE_INDEX)
                metric.update(shared_logits[None], shared_target[None])

    per_class_iou = metric.compute()
    mean_iou = per_class_iou.mean()
    return per_class_iou, mean_iou
```

## 6. Recommended workflow

For now:

- use the shared validation class for the Cityscapes-trained model
- use a standalone script for the COCO-trained model on Cityscapes
- once that works, you can decide whether to unify them

That keeps progress smooth and avoids fighting `main.py` too early.
