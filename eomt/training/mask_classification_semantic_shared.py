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
        num_shared_classes: int = 7,
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
            num_shared_classes=num_shared_classes,
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

        num_blocks = (
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        )
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

        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_layer, class_logits_per_layer)
        ):
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
