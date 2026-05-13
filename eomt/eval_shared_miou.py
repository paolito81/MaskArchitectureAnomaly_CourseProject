import argparse
import importlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.amp import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.datasets import Cityscapes

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
        nargs="+",
        help="One or more source model configs (e.g. COCO panoptic or Cityscapes semantic yaml).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        nargs="+",
        help="One or more checkpoints to evaluate. Must match the number of configs.",
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
        "--paradigms",
        nargs="+",
        choices=["semantic", "instance", "panoptic"],
        default=["semantic", "instance", "panoptic"],
        help="Inference paradigms to evaluate for each checkpoint.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of validation images to process for debugging.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "online", "offline"],
        default="disabled",
        help="Enable W&B logging for the standalone evaluator.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional W&B project override. Defaults to the config logger project if present.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional W&B run name override. Defaults to the config logger name if present.",
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


def get_cityscapes_stuff_classes(num_classes: int):
    stuff_classes = []
    for cls in Cityscapes.classes:
        if cls.ignore_in_eval or cls.train_id < 0 or cls.train_id >= num_classes:
            continue
        if not cls.has_instances:
            stuff_classes.append(cls.train_id)
    return sorted(set(stuff_classes))


def get_eval_model_class(paradigm: str):
    paradigm_to_class_path = {
        "semantic": "training.mask_classification_semantic.MaskClassificationSemantic",
        "instance": "training.mask_classification_instance.MaskClassificationInstance",
        "panoptic": "training.mask_classification_panoptic.MaskClassificationPanoptic",
    }
    return import_class(paradigm_to_class_path[paradigm])


def filter_kwargs_for_constructor(cls, kwargs: dict):
    signature = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def build_model_from_config(
    config: dict,
    masked_attn_enabled: bool,
    device: str,
    paradigm: str,
):
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

    model_init_args = config["model"]["init_args"]
    model_cls = get_eval_model_class(paradigm)
    model_kwargs = {k: v for k, v in model_init_args.items() if k != "network"}
    model_kwargs["attn_mask_annealing_enabled"] = model_init_args.get(
        "attn_mask_annealing_enabled", False
    )

    if paradigm == "panoptic":
        if "stuff_classes" in source_data_kwargs:
            model_kwargs["stuff_classes"] = source_data_kwargs["stuff_classes"]
        else:
            model_kwargs["stuff_classes"] = get_cityscapes_stuff_classes(
                source_num_classes
            )

    model_kwargs = filter_kwargs_for_constructor(model_cls, model_kwargs)

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


def update_confusion_matrix(
    confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
):
    valid = target != IGNORE_INDEX
    if not valid.any():
        return

    target_valid = target[valid].to(torch.int64)
    pred_valid = pred[valid].to(torch.int64)
    num_classes = confusion.shape[0]
    flat = target_valid * num_classes + pred_valid
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def compute_iou_from_confusion(confusion: torch.Tensor):
    tp = torch.diag(confusion)
    gt_support = confusion.sum(dim=1)
    pred_support = confusion.sum(dim=0)
    denom = gt_support + pred_support - tp
    iou = torch.where(denom > 0, tp.float() / denom.float(), torch.nan)
    return iou, gt_support, pred_support


def init_wandb_run(config: dict, args):
    if args.wandb_mode == "disabled":
        return None

    logger_cfg = config.get("trainer", {}).get("logger", {})
    init_args = logger_cfg.get("init_args", {})
    project = args.wandb_project or init_args.get("project", "eomt")
    name = args.wandb_name or init_args.get("name", "shared_eval")

    run = wandb.init(
        project=project,
        name=name,
        mode=args.wandb_mode,
        config={
            "config_paths": args.config,
            "checkpoints": args.ckpt,
            "cityscapes_path": args.cityscapes_path,
            "shared_classes": SHARED_CLASSES,
            "src_label_space": args.src_label_space,
            "masked_attn_enabled": args.masked_attn_enabled,
            "paradigms": args.paradigms,
        },
    )
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
    )
    return run


def make_example_figure(img: torch.Tensor, target: torch.Tensor, pred: torch.Tensor):
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
    target_np = target.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    unique_classes = np.unique(
        np.concatenate(
            [
                np.unique(target_np[target_np != IGNORE_INDEX]),
                np.unique(pred_np),
            ]
        )
    )
    colors = plt.get_cmap("tab20", max(len(unique_classes), 1))(
        np.linspace(0, 1, max(len(unique_classes), 1))
    )
    color_map = {cls_id: colors[i] for i, cls_id in enumerate(unique_classes)}
    color_map[IGNORE_INDEX] = np.array([0.0, 0.0, 0.0, 1.0])

    def colorize(mask):
        out = np.zeros((*mask.shape, 4), dtype=np.float32)
        for cls_id, color in color_map.items():
            out[mask == cls_id] = color
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[1].imshow(colorize(target_np))
    axes[1].set_title("Shared GT")
    axes[2].imshow(colorize(pred_np))
    axes[2].set_title("Shared Pred")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


def compute_shared_target(targets):
    per_pixel_targets = []

    for target in targets:
        per_pixel_target = target["labels"].new_full(
            target["masks"].shape[-2:],
            IGNORE_INDEX,
            device=target["labels"].device,
        )
        for i, mask in enumerate(target["masks"]):
            per_pixel_target[mask] = target["labels"][i]
        per_pixel_targets.append(
            remap_target_ids(per_pixel_target, CITYSCAPES_TO_SHARED, IGNORE_INDEX)
        )

    return per_pixel_targets


def predict_semantic_shared(model, imgs, src_to_shared):
    img_sizes = [img.shape[-2:] for img in imgs]
    crops, origins = model.window_imgs_semantic(imgs)
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
    logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return [
        remap_logits(logit, src_to_shared, len(SHARED_CLASSES)) for logit in logits
    ]


def predict_instance_shared(model, imgs, src_to_shared):
    img_sizes = [img.shape[-2:] for img in imgs]
    transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
    mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
    mask_logits = F.interpolate(
        mask_logits_per_layer[-1],
        model.img_size,
        mode="bilinear",
    )
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, img_sizes
    )

    shared_logits_list = []
    num_shared_classes = len(SHARED_CLASSES)
    for sample_idx in range(len(mask_logits)):
        scores = class_logits_per_layer[-1][sample_idx].softmax(dim=-1)[:, :-1]
        labels = (
            torch.arange(scores.shape[-1], device=model.device)
            .unsqueeze(0)
            .repeat(scores.shape[0], 1)
            .flatten(0, 1)
        )

        top_k = min(model.eval_top_k_instances, scores.numel())
        topk_scores, topk_indices = scores.flatten(0, 1).topk(top_k, sorted=False)
        labels = labels[topk_indices]
        topk_query_indices = topk_indices // scores.shape[-1]
        topk_mask_logits = mask_logits[sample_idx][topk_query_indices]

        shared_logits = topk_mask_logits.new_zeros(
            (num_shared_classes, *topk_mask_logits.shape[-2:])
        )
        for query_mask_logits, label_id, score in zip(
            topk_mask_logits,
            labels.tolist(),
            topk_scores.tolist(),
        ):
            if label_id not in src_to_shared:
                continue
            mask = query_mask_logits > 0
            if not mask.any():
                continue
            dst_id = src_to_shared[label_id]
            pixel_scores = query_mask_logits.sigmoid() * score
            shared_logits[dst_id] = torch.maximum(shared_logits[dst_id], pixel_scores)

        shared_logits_list.append(shared_logits)

    return shared_logits_list


def predict_panoptic_shared(model, imgs, src_to_shared):
    img_sizes = [img.shape[-2:] for img in imgs]
    transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
    mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
    mask_logits = F.interpolate(
        mask_logits_per_layer[-1],
        model.img_size,
        mode="bilinear",
    )
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, img_sizes
    )
    preds = model.to_per_pixel_preds_panoptic(
        mask_logits,
        class_logits_per_layer[-1],
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )

    shared_logits_list = []
    num_shared_classes = len(SHARED_CLASSES)
    for pred in preds:
        shared_logits = pred.new_zeros(
            (num_shared_classes, pred.shape[0], pred.shape[1]), dtype=torch.float32
        )
        class_ids = pred[:, :, 0]
        for src_id, dst_id in src_to_shared.items():
            shared_logits[dst_id][class_ids == src_id] = 1.0
        shared_logits_list.append(shared_logits)

    return shared_logits_list


def predict_shared_logits(model, imgs, src_to_shared, paradigm: str):
    if paradigm == "semantic":
        return predict_semantic_shared(model, imgs, src_to_shared)
    if paradigm == "instance":
        return predict_instance_shared(model, imgs, src_to_shared)
    if paradigm == "panoptic":
        return predict_panoptic_shared(model, imgs, src_to_shared)
    raise ValueError(f"Unsupported paradigm: {paradigm}")


def evaluate(
    model,
    loader,
    src_to_shared,
    device: str,
    paradigm: str,
    limit: int | None = None,
):
    num_shared_classes = len(SHARED_CLASSES)
    metric = MulticlassJaccardIndex(
        num_classes=num_shared_classes,
        average=None,
        ignore_index=IGNORE_INDEX,
    ).to(device)
    confusion = torch.zeros(
        (num_shared_classes, num_shared_classes), dtype=torch.int64, device=device
    )
    ignored_pixels = 0
    valid_pixels = 0
    example = None

    processed = 0
    use_autocast = str(device).startswith("cuda")

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = move_targets_to_device(targets, device)

            with autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_autocast
            ):
                shared_logits_list = predict_shared_logits(
                    model,
                    imgs,
                    src_to_shared,
                    paradigm,
                )
            per_pixel_targets = compute_shared_target(targets)

            for sample_idx, (shared_logits, shared_target) in enumerate(
                zip(shared_logits_list, per_pixel_targets)
            ):
                metric.update(shared_logits[None], shared_target[None])
                shared_pred = shared_logits.argmax(dim=0)
                shared_pred_vis = shared_pred.clone()
                shared_pred_vis[shared_target == IGNORE_INDEX] = IGNORE_INDEX

                update_confusion_matrix(confusion, shared_pred, shared_target)
                ignored_pixels += int((shared_target == IGNORE_INDEX).sum().item())
                valid_pixels += int((shared_target != IGNORE_INDEX).sum().item())
                if example is None:
                    example = {
                        "img": imgs[sample_idx].detach().cpu(),
                        "target": shared_target.detach().cpu(),
                        "pred": shared_pred_vis.detach().cpu(),
                    }
                processed += 1

                if limit is not None and processed >= limit:
                    break

            if limit is not None and processed >= limit:
                break

    per_class_iou = metric.compute()
    mean_iou = per_class_iou.mean()
    confusion = confusion.cpu()
    confusion_iou, gt_support, pred_support = compute_iou_from_confusion(confusion)
    return {
        "per_class_iou": per_class_iou.cpu(),
        "mean_iou": mean_iou.cpu(),
        "processed": processed,
        "confusion": confusion,
        "confusion_iou": confusion_iou,
        "gt_support": gt_support,
        "pred_support": pred_support,
        "ignored_pixels": ignored_pixels,
        "valid_pixels": valid_pixels,
        "example": example,
    }


def evaluate_checkpoint(config_path: str, ckpt_path: str, args, loader, wandb_run):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    src_to_shared = infer_source_mapping(config, args.src_label_space)
    config_stem = Path(config_path).stem
    ckpt_stem = Path(ckpt_path).stem

    print(f"\n=== Evaluating checkpoint: {ckpt_path} ===")
    print(f"Config: {config_path}")

    for paradigm in args.paradigms:
        model, source_num_classes, source_img_size = build_model_from_config(
            config=config,
            masked_attn_enabled=args.masked_attn_enabled,
            device=args.device,
            paradigm=paradigm,
        )
        load_checkpoint(model, ckpt_path)

        print(f"\n--- Paradigm: {paradigm} ---")
        print(f"Source model classes: {source_num_classes}")
        print(f"Source model img_size: {source_img_size}")
        print(f"Evaluating on Cityscapes val from: {args.cityscapes_path}")
        print(f"Shared classes ({len(SHARED_CLASSES)}): {SHARED_CLASSES}")

        results = evaluate(
            model=model,
            loader=loader,
            src_to_shared=src_to_shared,
            device=args.device,
            paradigm=paradigm,
            limit=args.limit,
        )

        per_class_iou = results["per_class_iou"]
        mean_iou = results["mean_iou"]
        confusion_iou = results["confusion_iou"]
        gt_support = results["gt_support"]
        pred_support = results["pred_support"]
        processed = results["processed"]
        total_pixels = results["ignored_pixels"] + results["valid_pixels"]
        ignored_ratio = (
            results["ignored_pixels"] / total_pixels
            if total_pixels > 0
            else float("nan")
        )

        print(f"Processed {processed} validation images")
        print(
            f"Valid pixels: {results['valid_pixels']:,} | "
            f"Ignored pixels: {results['ignored_pixels']:,} | "
            f"Ignored ratio: {ignored_ratio * 100:.2f}%"
        )
        print("Per-class IoU:")
        for class_name, iou in zip(SHARED_CLASSES, per_class_iou.tolist()):
            print(f"  {class_name:15s} {iou * 100:6.2f}")
        print("Per-class support and confusion-derived IoU:")
        for class_name, iou, gt_count, pred_count in zip(
            SHARED_CLASSES,
            confusion_iou.tolist(),
            gt_support.tolist(),
            pred_support.tolist(),
        ):
            iou_str = "nan" if iou != iou else f"{iou * 100:6.2f}"
            print(
                f"  {class_name:15s} IoU={iou_str:>6s} | "
                f"GT pixels={gt_count:9d} | Pred pixels={pred_count:9d}"
            )
        print(f"Shared mIoU: {mean_iou.item() * 100:.2f}")
        print("Confusion matrix (rows=GT, cols=Pred):")
        print(results["confusion"].numpy())

        if wandb_run is not None:
            prefix = f"{config_stem}/{ckpt_stem}/{paradigm}"
            log_dict = {
                f"{prefix}/metrics/shared_iou_all": mean_iou.item(),
                f"{prefix}/audit/valid_pixels": results["valid_pixels"],
                f"{prefix}/audit/ignored_pixels": results["ignored_pixels"],
                f"{prefix}/audit/ignored_ratio": ignored_ratio,
            }
            for idx, (class_name, iou, gt_count, pred_count) in enumerate(
                zip(
                    SHARED_CLASSES,
                    per_class_iou.tolist(),
                    gt_support.tolist(),
                    pred_support.tolist(),
                )
            ):
                safe_name = class_name.replace(" ", "_")
                log_dict[f"{prefix}/metrics/shared_iou_class_{idx}"] = iou
                log_dict[f"{prefix}/metrics/shared_iou/{safe_name}"] = iou
                log_dict[f"{prefix}/audit/gt_pixels/{safe_name}"] = gt_count
                log_dict[f"{prefix}/audit/pred_pixels/{safe_name}"] = pred_count
            wandb.log(log_dict)

            if results["example"] is not None:
                fig = make_example_figure(
                    results["example"]["img"],
                    results["example"]["target"],
                    results["example"]["pred"],
                )
                wandb.log(
                    {f"{prefix}/qualitative/shared_eval_example": wandb.Image(fig)}
                )
                plt.close(fig)


def main():
    args = parse_args()
    if len(args.config) != len(args.ckpt):
        raise ValueError(
            f"--config and --ckpt must have the same length, got "
            f"{len(args.config)} configs and {len(args.ckpt)} checkpoints."
        )

    with open(args.config[0], "r") as f:
        first_config = yaml.safe_load(f)

    wandb_run = init_wandb_run(first_config, args)
    loader = build_cityscapes_loader(
        cityscapes_path=args.cityscapes_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for config_path, ckpt_path in zip(args.config, args.ckpt):
        evaluate_checkpoint(config_path, ckpt_path, args, loader, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
