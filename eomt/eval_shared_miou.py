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
from matplotlib.patches import Patch
from torch.amp import autocast
from torchmetrics.classification import MulticlassJaccardIndex

from datasets.cityscapes_semantic import CityscapesSemantic
from shared_eval.shared import (
    CITYSCAPES_CLASSES,
    CITYSCAPES_LABEL_TO_ID,
    CITYSCAPES_TO_CITYSCAPES,
    CITYSCAPES_TO_SHARED,
    COCO_TO_CITYSCAPES,
    COCO_TO_SHARED,
    IGNORE_INDEX,
    SHARED_CLASSES,
    remap_logits,
    remap_target_ids,
)

SHARED_CLASS_COLORS = {
    0: np.array([0.89, 0.10, 0.11, 1.0]),  # person
    1: np.array([0.12, 0.47, 0.71, 1.0]),  # car
    2: np.array([1.00, 0.50, 0.05, 1.0]),  # truck
    3: np.array([0.17, 0.63, 0.17, 1.0]),  # bus
    4: np.array([0.58, 0.40, 0.74, 1.0]),  # motorcycle
    5: np.array([0.55, 0.34, 0.29, 1.0]),  # bicycle
    6: np.array([0.84, 0.15, 0.16, 1.0]),  # traffic light
}
IGNORE_COLOR = np.array([0.0, 0.0, 0.0, 1.0])


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
        "--eval-label-space",
        choices=["shared", "cityscapes"],
        default="shared",
        help="Label space used for metrics. 'cityscapes' evaluates in the 19-class Cityscapes space.",
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
    parser.add_argument(
        "--example-min-classes",
        type=int,
        default=4,
        help="Minimum number of evaluated ground-truth classes required before an image is kept as the qualitative example.",
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


def infer_source_space(config, override: str):
    if override != "auto":
        return override

    data_class_path = config["data"]["class_path"]
    if "coco_panoptic" in data_class_path:
        return "coco"
    if "cityscapes_semantic" in data_class_path:
        return "cityscapes"
    raise ValueError(
        f"Could not infer source label space from data class path: {data_class_path}"
    )


def get_eval_spec(source_space: str, eval_label_space: str):
    if eval_label_space == "shared":
        src_to_eval = (
            COCO_TO_SHARED if source_space == "coco" else CITYSCAPES_TO_SHARED
        )
        return {
            "name": "shared",
            "class_names": SHARED_CLASSES,
            "src_to_eval": src_to_eval,
            "target_to_eval": CITYSCAPES_TO_SHARED,
            "colors": SHARED_CLASS_COLORS,
        }

    if eval_label_space == "cityscapes":
        src_to_eval = (
            COCO_TO_CITYSCAPES
            if source_space == "coco"
            else CITYSCAPES_TO_CITYSCAPES
        )
        target_to_eval = {
            cityscapes_id: cityscapes_id
            for cityscapes_id in set(src_to_eval.values())
        }
        return {
            "name": "cityscapes",
            "class_names": CITYSCAPES_CLASSES,
            "src_to_eval": src_to_eval,
            "target_to_eval": target_to_eval,
            "colors": None,
        }

    raise ValueError(f"Unsupported eval label space: {eval_label_space}")


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


def normalize_confusion_rows(confusion: torch.Tensor):
    row_sums = confusion.sum(dim=1, keepdim=True)
    return torch.where(
        row_sums > 0,
        confusion.float() / row_sums.float(),
        torch.zeros_like(confusion, dtype=torch.float32),
    )


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
            "config_path": args.config,
            "checkpoint": args.ckpt,
            "cityscapes_path": args.cityscapes_path,
            "eval_label_space": args.eval_label_space,
            "src_label_space": args.src_label_space,
            "masked_attn_enabled": args.masked_attn_enabled,
        },
    )
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
    )
    return run


def make_example_figure(
    img: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    class_names: list[str],
    colors: dict[int, np.ndarray] | None = None,
    title_prefix: str = "",
):
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
    color_map = {}
    palette = plt.cm.get_cmap("tab20", len(class_names))
    for cls_id in unique_classes:
        if int(cls_id) == IGNORE_INDEX:
            color_map[cls_id] = IGNORE_COLOR
        else:
            if colors is not None and int(cls_id) in colors:
                color_map[cls_id] = colors[int(cls_id)]
            else:
                color_map[cls_id] = palette(int(cls_id))

    def colorize(mask):
        out = np.zeros((*mask.shape, 4), dtype=np.float32)
        for cls_id, color in color_map.items():
            out[mask == cls_id] = color
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[1].imshow(colorize(target_np))
    axes[1].set_title(f"{title_prefix}GT".strip())
    axes[2].imshow(colorize(pred_np))
    axes[2].set_title(f"{title_prefix}Pred".strip())
    for ax in axes:
        ax.axis("off")
    legend_handles = []
    for cls_id in unique_classes:
        if cls_id == IGNORE_INDEX:
            label = "ignored"
        else:
            label = class_names[int(cls_id)]
        legend_handles.append(
            Patch(facecolor=color_map[cls_id], edgecolor="black", label=label)
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def count_valid_shared_classes(mask: torch.Tensor):
    valid_classes = torch.unique(mask[mask != IGNORE_INDEX])
    return int(valid_classes.numel())


def evaluate(
    model,
    loader,
    eval_spec,
    device: str,
    limit: int | None = None,
    example_min_classes: int = 4,
):
    class_names = eval_spec["class_names"]
    src_to_eval = eval_spec["src_to_eval"]
    target_to_eval = eval_spec["target_to_eval"]
    num_eval_classes = len(class_names)
    metric = MulticlassJaccardIndex(
        num_classes=num_eval_classes,
        average=None,
        ignore_index=IGNORE_INDEX,
    ).to(device)
    confusion = torch.zeros(
        (num_eval_classes, num_eval_classes), dtype=torch.int64, device=device
    )
    ignored_pixels = 0
    valid_pixels = 0
    example = None
    best_example = None

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

            for sample_idx, (logit, target) in enumerate(
                zip(logits, per_pixel_targets)
            ):
                eval_logits = remap_logits(logit, src_to_eval, num_eval_classes)
                eval_target = remap_target_ids(target, target_to_eval, IGNORE_INDEX)
                metric.update(eval_logits[None], eval_target[None])
                eval_pred = eval_logits.argmax(dim=0)
                eval_pred_vis = eval_pred.clone()
                eval_pred_vis[eval_target == IGNORE_INDEX] = IGNORE_INDEX

                update_confusion_matrix(confusion, eval_pred, eval_target)
                ignored_pixels += int((eval_target == IGNORE_INDEX).sum().item())
                valid_pixels += int((eval_target != IGNORE_INDEX).sum().item())
                num_present_classes = count_valid_shared_classes(eval_target)
                candidate_example = {
                    "img": imgs[sample_idx].detach().cpu(),
                    "target": eval_target.detach().cpu(),
                    "pred": eval_pred_vis.detach().cpu(),
                    "num_present_classes": num_present_classes,
                }
                if (
                    best_example is None
                    or num_present_classes > best_example["num_present_classes"]
                ):
                    best_example = candidate_example
                if example is None and num_present_classes >= example_min_classes:
                    example = candidate_example
                processed += 1

                if limit is not None and processed >= limit:
                    break

            if limit is not None and processed >= limit:
                break

    if example is None:
        example = best_example

    per_class_iou = metric.compute()
    mean_iou = per_class_iou.mean()
    confusion = confusion.cpu()
    confusion_iou, gt_support, pred_support = compute_iou_from_confusion(confusion)
    confusion_normalized = normalize_confusion_rows(confusion)
    return {
        "per_class_iou": per_class_iou.cpu(),
        "mean_iou": mean_iou.cpu(),
        "processed": processed,
        "confusion": confusion,
        "confusion_normalized": confusion_normalized,
        "confusion_iou": confusion_iou,
        "gt_support": gt_support,
        "pred_support": pred_support,
        "ignored_pixels": ignored_pixels,
        "valid_pixels": valid_pixels,
        "example": example,
    }


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb_run = init_wandb_run(config, args)
    source_space = infer_source_space(config, args.src_label_space)
    eval_spec = get_eval_spec(source_space, args.eval_label_space)
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
    print(f"Source label space: {source_space}")
    print(f"Evaluating on Cityscapes val from: {args.cityscapes_path}")
    print(
        f"Evaluation label space: {eval_spec['name']} "
        f"({len(eval_spec['class_names'])} classes)"
    )
    print(f"Evaluation classes: {eval_spec['class_names']}")
    if args.eval_label_space == "cityscapes" and source_space == "coco":
        mapped_cityscapes = sorted(set(eval_spec["target_to_eval"].keys()))
        mapped_names = [CITYSCAPES_CLASSES[i] for i in mapped_cityscapes]
        print(
            "Note: COCO -> Cityscapes evaluation only scores mapped Cityscapes classes "
            f"and ignores unmatched GT classes: {mapped_names}"
        )

    results = evaluate(
        model=model,
        loader=loader,
        eval_spec=eval_spec,
        device=args.device,
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
        results["ignored_pixels"] / total_pixels if total_pixels > 0 else float("nan")
    )

    print(f"\nProcessed {processed} validation images")
    print(
        f"Valid pixels: {results['valid_pixels']:,} | "
        f"Ignored pixels: {results['ignored_pixels']:,} | "
        f"Ignored ratio: {ignored_ratio * 100:.2f}%"
    )
    print("Per-class IoU:")
    for class_name, iou in zip(eval_spec["class_names"], per_class_iou.tolist()):
        print(f"  {class_name:15s} {iou * 100:6.2f}")
    print("\nPer-class support and confusion-derived IoU:")
    for class_name, iou, gt_count, pred_count in zip(
        eval_spec["class_names"],
        confusion_iou.tolist(),
        gt_support.tolist(),
        pred_support.tolist(),
    ):
        iou_str = "nan" if iou != iou else f"{iou * 100:6.2f}"
        print(
            f"  {class_name:15s} IoU={iou_str:>6s} | "
            f"GT pixels={gt_count:9d} | Pred pixels={pred_count:9d}"
        )
    print(f"\n{eval_spec['name'].title()} mIoU: {mean_iou.item() * 100:.2f}")
    print("\nConfusion matrix (rows=GT, cols=Pred):")
    print(results["confusion"].numpy())
    print("\nNormalized confusion matrix (rows=GT, each row sums to 1):")
    print(results["confusion_normalized"].numpy())

    if wandb_run is not None:
        metric_prefix = f"metrics/{eval_spec['name']}_iou"
        log_dict = {
            f"{metric_prefix}_all": mean_iou.item(),
            "audit/valid_pixels": results["valid_pixels"],
            "audit/ignored_pixels": results["ignored_pixels"],
            "audit/ignored_ratio": ignored_ratio,
        }
        for idx, (class_name, iou, gt_count, pred_count) in enumerate(
            zip(
                eval_spec["class_names"],
                per_class_iou.tolist(),
                gt_support.tolist(),
                pred_support.tolist(),
            )
        ):
            safe_name = class_name.replace(" ", "_")
            log_dict[f"{metric_prefix}_class_{idx}"] = iou
            log_dict[f"{metric_prefix}/{safe_name}"] = iou
            log_dict[f"audit/gt_pixels/{safe_name}"] = gt_count
            log_dict[f"audit/pred_pixels/{safe_name}"] = pred_count
        wandb.log(log_dict)

        if results["example"] is not None:
            fig = make_example_figure(
                results["example"]["img"],
                results["example"]["target"],
                results["example"]["pred"],
                class_names=eval_spec["class_names"],
                colors=eval_spec["colors"],
                title_prefix=f"{eval_spec['name'].title()} ",
            )
            wandb.log(
                {
                    f"qualitative/{eval_spec['name']}_eval_example": wandb.Image(fig)
                }
            )
            plt.close(fig)

        wandb_run.finish()


if __name__ == "__main__":
    main()
