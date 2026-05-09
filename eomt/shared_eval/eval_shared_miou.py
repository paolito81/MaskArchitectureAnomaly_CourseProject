import torch
from torchmetrics.classification import MulticlassJaccardIndex

from shared_eval.shared import (
    SHARED_CLASSES,
    IGNORE_INDEX,
    remap_logits,
    remap_target_ids,
    CITYSCAPES_TO_SHARED,
    COCO_TO_SHARED,
)


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
            logits = model.revert_window_logits_semantic(
                crop_logits, origins, img_sizes
            )

            per_pixel_targets = model.to_per_pixel_targets_semantic(
                targets, IGNORE_INDEX
            )

            for logit, target in zip(logits, per_pixel_targets):
                shared_logits = remap_logits(logit, src_to_shared, len(SHARED_CLASSES))
                shared_target = remap_target_ids(
                    target, CITYSCAPES_TO_SHARED, IGNORE_INDEX
                )
                metric.update(shared_logits[None], shared_target[None])

    per_class_iou = metric.compute()
    mean_iou = per_class_iou.mean()
    return per_class_iou, mean_iou
