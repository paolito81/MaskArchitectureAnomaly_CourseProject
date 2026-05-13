# Anomaly segmentation pixel-based baselines for ERFNet.
# Implements three post-hoc uncertainty methods:
#   - MSP          (Maximum Softmax Probability)
#   - MaxLogit     (Maximum Raw Logit)
#   - MaxEntropy   (Shannon Entropy over softmax distribution)
#
# Usage:
#   python evalAnomaly_methods.py --method msp       --input <path_to_images_folder>
#   python evalAnomaly_methods.py --method maxlogit  --input <path_to_images_folder>
#   python evalAnomaly_methods.py --method maxentropy --input <path_to_images_folder>
#

import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr

from erfnet import ERFNet

# ──────────────────────────────────────────────────────────────
# Reproducibility seeds
# ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False      # Ensures reproducibility at the cost of some performance.

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
NUM_CLASSES = 20
IGNORE_LABEL = 255

# ──────────────────────────────────────────────────────────────
# Image & mask transforms
# ──────────────────────────────────────────────────────────────
# Resize input images to 512x1024 (ERFNet's training resolution)
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),                          # → [C, H, W], float32 in [0, 1]
])

# Resize GT masks with nearest-neighbour to preserve label values
target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


# ──────────────────────────────────────────────────────────────
# Anomaly score functions
# Each function receives logits of shape [1, C, H, W] (on GPU)
# and returns a numpy array of shape [H, W] (higher = more anomalous)
# ──────────────────────────────────────────────────────────────

def compute_msp(logits: torch.Tensor) -> np.ndarray:
    """
    MSP — Maximum Softmax Probability.

    Formula:  score(x) = 1 - max_c( softmax(logits)_c )

    Intuition:
        A confident model concentrates probability on one class → high max → low score.
        An uncertain model spreads probability → low max → high score → anomaly.

    Note:
        ERFNet returns raw logits (no softmax inside the model).
        We MUST apply softmax here before taking the max.
        Without softmax this would compute 1-max(logits) which is
        unbounded and not a valid probability-based score.

    Args:
        logits: raw model output, shape [1, C, H, W], on GPU
    Returns:
        anomaly_map: numpy array [H, W], higher = more anomalous
    """
    # Apply softmax over class dimension → valid probability distribution
    probs = torch.softmax(logits, dim=1)          # [1, C, H, W]

    # Take the maximum probability across all classes per pixel
    max_prob = probs.max(dim=1).values            # [1, H, W]

    # Anomaly score: 1 - confidence
    anomaly_map = 1.0 - max_prob.squeeze(0)       # [H, W]

    return anomaly_map.cpu().numpy()


def compute_maxlogit(logits: torch.Tensor) -> np.ndarray:
    """
    MaxLogit — Maximum Raw Logit score.

    Formula:  score(x) = -max_c( logits_c )

    Intuition:
        Raw logits carry magnitude information that softmax destroys.
        For known classes the model produces large positive logits.
        For unknown objects the max logit is smaller → higher anomaly score.

    Note:
        No softmax needed here. We use logits directly.
        We negate so that higher score = more anomalous (consistent with MSP).

    Args:
        logits: raw model output, shape [1, C, H, W], on GPU
    Returns:
        anomaly_map: numpy array [H, W], higher = more anomalous
    """
    max_logit = logits.max(dim=1).values          # [1, H, W]
    anomaly_map = -max_logit.squeeze(0)           # [H, W], negated

    return anomaly_map.cpu().numpy()


def compute_max_entropy(logits: torch.Tensor) -> np.ndarray:
    """
    MaxEntropy — Normalized Shannon entropy of the softmax distribution.

    Formula:   H(x) = -sum_c [ p_c * log(p_c) ] / log(C)
               score(x) = H(x)

    Intuition:
        Entropy measures how "spread" the probability distribution is.
        Normalized to [0, 1] by dividing by the maximum possible entropy (log(C)).
        
        Known object   → concentrated distribution → entropy ~0 → low score.
        Unknown object → spread/uniform distribution → entropy ~1 → high score.

    Numerical stability:
        log(0) is undefined. We add a small epsilon (1e-10) inside the log
        to prevent NaN/Inf values when any probability approaches zero.

    Args:
        logits: raw model output, shape [1, C, H, W], on GPU
    Returns:
        anomaly_map: numpy array [H, W], range [0, 1], higher = more anomalous
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)                      # [1, C, H, W]

    # Compute per-pixel entropy: H = -sum(p * log(p))
    # epsilon added for numerical stability
    log_probs = torch.log(probs + 1e-10)                      # [1, C, H, W]
    entropy = -(probs * log_probs).sum(dim=1)                 # [1, H, W]

    # NORMALIZATION: Divide by log(C) to bound the result between 0 and 1
    normalized_entropy = entropy / torch.log(torch.tensor(NUM_CLASSES, device=entropy.device))

    anomaly_map = normalized_entropy.squeeze(0)                          # [H, W]

    return anomaly_map.cpu().numpy()


# ──────────────────────────────────────────────────────────────
# Method registry — maps CLI name to function
# ──────────────────────────────────────────────────────────────
METHODS = {
    "msp":        compute_msp,
    "maxlogit":   compute_maxlogit,
    "maxentropy": compute_max_entropy,
}


# ──────────────────────────────────────────────────────────────
# Ground truth loading and label remapping
# ──────────────────────────────────────────────────────────────

def get_gt_path(image_path: str) -> str:
    """
    Build the ground truth mask path from an image path.

    Datasets store masks in 'labels_masks/' next to 'images/'.
    Some datasets also require an extension change (e.g. .webp → .png).

    Args:
        image_path: full path to input image
    Returns:
        path to corresponding GT mask (.png)
    """
    # Replace 'images' subfolder with 'labels_masks'
    gt_path = image_path.replace("images", "labels_masks")

    # RoadObsticle21 images are .webp but masks are .png
    if "RoadObsticle21" in gt_path:
        gt_path = gt_path.replace(".webp", ".png")

    # fs_static images are .jpg but masks are .png
    if "fs_static" in gt_path:
        gt_path = gt_path.replace(".jpg", ".png")

    # RoadAnomaly images are .jpg but masks are .png
    if "RoadAnomaly" in gt_path and not "RoadAnomaly21" in gt_path:
        gt_path = gt_path.replace(".jpg", ".png")

    return gt_path


def load_gt_mask(gt_path: str) -> np.ndarray:
    """
    Load and remap a ground truth mask to standard binary format:
        0   = in-distribution (normal road scene)
        1   = out-of-distribution (anomaly)
        255 = ignore (excluded from metric computation)

    Dataset-specific label conventions:
        - RoadAnomaly        : 0/1 = normal, 2 = anomaly      → remap 2 → 1
        - RoadAnomaly21      : 0 = normal, 1 = anomaly, 255 = ignore (standard)
        - RoadObsticle21     : 0 = normal, 1 = anomaly, 255 = ignore (standard)
        - FS_LostFound_full  : 0 = normal, 1 = anomaly, 255 = ignore (standard)
        - fs_static          : 0 = normal, 1 = anomaly, 255 = ignore (standard)

    Args:
        gt_path: path to the GT mask file
    Returns:
        ood_gts: numpy array [H, W] with values in {0, 1, 255}
    """
    mask = Image.open(gt_path)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    # Only RoadAnomaly (older standalone dataset) needs remapping
    # RoadAnomaly21 must be excluded from the check because it uses standard format
    if "RoadAnomaly" in gt_path and "RoadAnomaly21" not in gt_path:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)

    return ood_gts


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(weights_path: str) -> torch.nn.Module:
    """
    Load pretrained ERFNet and move to GPU.

    Uses a custom state dict loader to handle the 'module.' prefix
    that appears when weights were saved from a DataParallel model.

    Args:
        weights_path: path to .pth weights file
    Returns:
        model in eval mode on GPU
    """
    model = ERFNet(NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(f"  [SKIP] {name}")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    state_dict = torch.load(
        weights_path,
        map_location=lambda storage, loc: storage
    )
    model = load_my_state_dict(model, state_dict)
    print(f"[OK] Weights loaded from: {weights_path}")

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────

def evaluate(args):
    score_fn = METHODS[args.method]

    print(f"\n{'='*60}")
    print(f"  Method  : {args.method.upper()}")
    print(f"  Input   : {args.input}")
    print(f"{'='*60}\n")

    # Load model
    weights_path = os.path.join(args.loadDir, args.loadWeights)
    model = load_model(weights_path)

    # Collect all image paths (support .png, .jpg, .webp)
    input_dir = str(args.input)
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"[ERROR] No images found in: {input_dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images\n")

    anomaly_score_list = []
    ood_gts_list = []

    for path in image_paths:
        print(f"  {os.path.basename(path)}")

        # Load image → tensor → GPU
        img = Image.open(path).convert("RGB")
        img_tensor = input_transform(img).unsqueeze(0).float().cuda()
        # Shape: [1, 3, 512, 1024]
        # Note: ToTensor() already produces [C, H, W] — no permute needed

        # Forward pass (inference only — no gradient computation)
        with torch.no_grad():
            logits = model(img_tensor)          # [1, 20, 512, 1024] raw logits

        # Compute anomaly score map using selected method
        anomaly_map = score_fn(logits)          # [512, 1024] numpy float

        # Get and validate GT path
        gt_path = get_gt_path(path)
        if not os.path.exists(gt_path):
            print(f"    [WARN] GT not found: {gt_path} — skipping")
            del logits
            torch.cuda.empty_cache()
            continue

        # Load and remap GT mask
        ood_gts = load_gt_mask(gt_path)

        # Skip images that have no anomaly pixels (label 1)
        if 1 not in np.unique(ood_gts):
            del logits
            torch.cuda.empty_cache()
            continue

        anomaly_score_list.append(anomaly_map)
        ood_gts_list.append(ood_gts)

        del logits
        torch.cuda.empty_cache()

    if not ood_gts_list:
        print("[ERROR] No valid images with anomaly labels found.")
        return

    # ──────────────────────────────────────────────────────
    # Compute metrics
    # ──────────────────────────────────────────────────────
    ood_gts_arr = np.array(ood_gts_list)              # [N, H, W]
    scores_arr  = np.array(anomaly_score_list)         # [N, H, W]

    # Separate anomaly pixels (1) and normal pixels (0)
    # Pixels with label 255 (ignore) are automatically excluded
    # because they are neither == 1 nor == 0
    ood_mask = (ood_gts_arr == 1)
    ind_mask = (ood_gts_arr == 0)

    ood_scores = scores_arr[ood_mask]
    ind_scores = scores_arr[ind_mask]

    # Build flat arrays for metric computation
    val_scores = np.concatenate([ind_scores, ood_scores])
    val_labels = np.concatenate([
        np.zeros(len(ind_scores)),   # 0 = normal
        np.ones(len(ood_scores))     # 1 = anomaly
    ])

    auprc = average_precision_score(val_labels, val_scores)
    fpr95 = fpr_at_95_tpr(val_scores, val_labels)

    # ──────────────────────────────────────────────────────
    # Print and save results
    # ──────────────────────────────────────────────────────
    dataset_name = os.path.basename(os.path.dirname(input_dir.rstrip("/\\")))
    print(f"\n{'='*60}")
    print(f"  Method  : {args.method.upper()}")
    print(f"  Dataset : {dataset_name}")
    print(f"  AUPRC   : {auprc * 100:.2f}%")
    print(f"  FPR@95  : {fpr95 * 100:.2f}%")
    print(f"{'='*60}\n")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
    with open(results_path, "a") as f:
        f.write(
            f"\n{args.method.upper():12s} | {dataset_name:20s} | "
            f"AUPRC: {auprc*100:.2f}% | FPR@95: {fpr95*100:.2f}%"
        )
    print(f"[OK] Results saved to: {results_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(
        description="ERFNet anomaly segmentation — MSP / MaxLogit / MaxEntropy"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to dataset images folder"
    )
    parser.add_argument(
        "--method", required=True,
        choices=["msp", "maxlogit", "maxentropy"],
        help="Anomaly scoring method"
    )
    parser.add_argument(
        "--loadDir", default="../trained_models/",
        help="Directory containing model weights"
    )
    parser.add_argument(
        "--loadWeights", default="erfnet_pretrained.pth",
        help="Weights filename"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()