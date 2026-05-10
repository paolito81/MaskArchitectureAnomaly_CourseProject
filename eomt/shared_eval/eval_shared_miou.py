import argparse
import importlib
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torchmetrics.classification import MulticlassJaccardIndex

from datasets.cityscapes_semantic import CityscapesSemantic
from shared_eval.shared import (
    CITYSCAPES_TO_SHARED,
    COCO_TO_SHARED,
    IGNORE_INDEX,
    SHARED_CLASSES,
    remap_logits,
    remap_target_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an EoMT checkpoint on Cityscapes val using a shared label space."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the source model config (e.g. COCO panoptic or Cityscapes semantic yaml).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--cityscapes-path",
        required=True,
        help="Path to the Cityscapes dataset root containing the trainval zips.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--masked-attn-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override masked attention at inference time. Defaults to false to match notebook validation.",
    )
    parser.add_argument(
        "--src-label-space",
        choices=["auto", "coco", "cityscapes"],
        default="auto",
        help="Which mapping to use for model output channels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of validation images to process for debugging.",
    )
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_constructor_default(cls, param_name):
    signature = inspect.signature(cls.__init__)
    param = signature.parameters[param_name]
    if param.default is inspect._empty:
        raise ValueError(f"{cls.__name__}.__init__ has no default for {param_name}")
    return param.default


def infer_source_mapping(config, override: str):
    if override != "auto":
        return COCO_TO_SHARED if override == "coco" else CITYSCAPES_TO_SHARED

    data_class_path = config["data"]["class_path"]
    if "coco_panoptic" in data_class_path:
        return COCO_TO_SHARED
    if "cityscapes_semantic" in data_class_path:
        return CITYSCAPES_TO_SHARED
    raise ValueError(
        f"Could not infer source label space from data class path: {data_class_path}"
    )


def build_cityscapes_loader(cityscapes_path: str, batch_size: int, num_workers: int):
    data = CityscapesSemantic(
        path=cityscapes_path,
        batch_size=batch_size,
        num_workers=num_workers,
        check_empty_targets=False,
    ).setup()
    return data.val_dataloader()


def build_model_from_config(config: dict, masked_attn_enabled: bool, device: str):
    source_data_cls = import_class(config["data"]["class_path"])
    source_data_kwargs = config["data"].get("init_args", {})
    source_num_classes = source_data_kwargs.get(
        "num_classes", get_constructor_default(source_data_cls, "num_classes")
    )
    source_img_size = tuple(
        source_data_kwargs.get(
            "img_size", get_constructor_default(source_data_cls, "img_size")
        )
    )

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_cls = import_class(encoder_cfg["class_path"])
    encoder = encoder_cls(
        img_size=source_img_size,
        **encoder_cfg.get("init_args", {}),
    )

    network_cfg = config["model"]["init_args"]["network"]
    network_cls = import_class(network_cfg["class_path"])
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }
    network = network_cls(
        encoder=encoder,
        num_classes=source_num_classes,
        masked_attn_enabled=masked_attn_enabled,
        **network_kwargs,
    )

    model_cfg = config["model"]
    model_cls = import_class(model_cfg["class_path"])
    model_kwargs = {k: v for k, v in model_cfg["init_args"].items() if k != "network"}

    if "stuff_classes" in source_data_kwargs:
        model_kwargs["stuff_classes"] = source_data_kwargs["stuff_classes"]

    model = model_cls(
        network=network,
        img_size=source_img_size,
        num_classes=source_num_classes,
        **model_kwargs,
    ).eval()

    return model.to(device), source_num_classes, source_img_size


def load_checkpoint(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
    incompatible = model.load_state_dict(ckpt, strict=False)
    if incompatible.missing_keys:
        print(
            f"Missing keys while loading checkpoint: {len(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        print(
            f"Unexpected keys while loading checkpoint: {len(incompatible.unexpected_keys)}"
        )


def move_targets_to_device(targets, device: str):
    moved = []
    for target in targets:
        moved.append({k: v.to(device) for k, v in target.items()})
    return moved


def evaluate(model, loader, src_to_shared, device: str, limit: int | None = None):
    metric = MulticlassJaccardIndex(
        num_classes=len(SHARED_CLASSES),
        average=None,
        ignore_index=IGNORE_INDEX,
    ).to(device)

    processed = 0
    use_autocast = str(device).startswith("cuda")

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = move_targets_to_device(targets, device)

            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)

            with autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_autocast
            ):
                mask_logits_per_layer, class_logits_per_layer = model(crops)
                mask_logits = F.interpolate(
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
                processed += 1

                if limit is not None and processed >= limit:
                    break

            if limit is not None and processed >= limit:
                break

    per_class_iou = metric.compute()
    mean_iou = per_class_iou.mean()
    return per_class_iou, mean_iou, processed


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    src_to_shared = infer_source_mapping(config, args.src_label_space)
    loader = build_cityscapes_loader(
        cityscapes_path=args.cityscapes_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, source_num_classes, source_img_size = build_model_from_config(
        config=config,
        masked_attn_enabled=args.masked_attn_enabled,
        device=args.device,
    )
    load_checkpoint(model, args.ckpt)

    print(f"Loaded model from: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Source model classes: {source_num_classes}")
    print(f"Source model img_size: {source_img_size}")
    print(f"Evaluating on Cityscapes val from: {args.cityscapes_path}")
    print(f"Shared classes ({len(SHARED_CLASSES)}): {SHARED_CLASSES}")

    per_class_iou, mean_iou, processed = evaluate(
        model=model,
        loader=loader,
        src_to_shared=src_to_shared,
        device=args.device,
        limit=args.limit,
    )

    print(f"\nProcessed {processed} validation images")
    print("Per-class IoU:")
    for class_name, iou in zip(SHARED_CLASSES, per_class_iou.tolist()):
        print(f"  {class_name:15s} {iou * 100:6.2f}")
    print(f"\nShared mIoU: {mean_iou.item() * 100:.2f}")


if __name__ == "__main__":
    main()
