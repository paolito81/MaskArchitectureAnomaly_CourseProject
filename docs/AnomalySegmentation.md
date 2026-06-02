# Anomaly Segmentation — Theory, Datasets, and Post-Hoc Methods

**Project:** Comprehensive Road Scene Understanding for Autonomous Driving  
**Step:** 6 — Understand anomaly segmentation task and post-hoc methods  
**References:** [1] SegmentMeIfYouCan · [2] Fishyscapes · [7] RbA · [8] Scaling OOD Detection

---

## 1. Problem Definition

### 1.1 Standard Semantic Segmentation (Closed-World)

In standard semantic segmentation, a model is trained on a fixed set of N classes.
At test time, the assumption is that every pixel belongs to one of those N classes.
The model outputs a probability distribution over classes per pixel and assigns the
most likely class label.

This is called the **closed-world assumption**: the world only contains what we trained on.

### 1.2 The Open-World Problem

In the real world — especially in autonomous driving — a vehicle will encounter
objects that were never part of the training distribution. A cow on a highway.
A fallen tree. A child's toy on a road. These objects are called:

- **Out-of-Distribution (OoD)** — not seen during training
- **Anomalies** — statistically unexpected given the training data
- **Unknowns** — belonging to categories not in the label set

**Anomaly segmentation** is the task of producing a dense pixel-level anomaly map
alongside standard segmentation, identifying *where* in the image the unknown
objects are located.

### 1.3 Why Softmax Fails for OoD Detection

Neural networks with softmax output always produce probabilities that sum to 1.
Even when the input is completely alien to the model, it will confidently assign
that pixel to whichever known class it resembles most.

**Example with ERFNet (19 Cityscapes classes):**

```
Input:  a brown bear sitting on a road
Output: car=0.71, person=0.19, road=0.10
```

The model is confidently wrong. This is called **softmax overconfidence** and it
is a fundamental structural property of cross-entropy trained classifiers.

> **Key insight:** The model cannot output "I don't know" — it is forced to
> distribute 100% of probability across known classes. Post-hoc methods
> try to detect *when* this forced confidence is suspicious.

### 1.4 Image-Level vs Pixel-Level Anomaly Detection

| Level | Output | Granularity | Difficulty |
|---|---|---|---|
| Image-level | Single binary label | Whole image | Easier |
| Pixel-level (anomaly segmentation) | Dense map [H, W] | Per pixel | Much harder |

This project works at **pixel level** — every pixel needs an anomaly score.

### 1.5 Post-Hoc vs Training-Based Methods

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **Post-hoc** | Apply to any pretrained model, no retraining | Fast, model-agnostic | Limited by model |
| **Training-based** | Modify training with OoD data/loss | Stronger performance | Requires retraining |

We use **post-hoc methods** in this project, applied on top of a pretrained ERFNet.

---

## 2. Datasets

### 2.1 SegmentMeIfYouCan (SMIYC) — Reference [1]

**Paper:** Blum et al., *SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation*, 2021  
**Link:** https://arxiv.org/abs/2104.14812

SMIYC is a benchmark specifically designed for anomaly segmentation in road scenes.
It contains two evaluation tracks:

#### RoadAnomaly21 (RA-21)
- **Content:** Real road scene images with unusual, unexpected objects
- **Anomalies include:** animals (cows, horses, dogs), debris, unusual obstacles
- **Images:** Collected from the web — diverse lighting, weather, road types
- **Labels:** Binary mask
  - `0` = normal (road, cars, sky, vegetation — expected road scene elements)
  - `1` = anomaly (the unusual object)
- **Challenge:** High diversity of anomaly appearance; no two anomalies look the same

#### RoadObsticle21 (RO-21)
- **Content:** Real road scenes with specific obstacle types
- **Anomalies include:** wooden boxes, bags, logs, construction items placed on road
- **Labels:** Binary mask (same format as RA-21)
- **Challenge:** Obstacles can be small and texturally similar to road surface
- **Image format:** `.webp` (unusual format — requires special handling in loader)

#### How Labels Are Used in This Project

```
labels_masks/
    0001.png   ← grayscale, pixel values: 0 (normal) or 1 (anomaly)
    0002.png
    ...
```

For RoadAnomaly21: pixel value `1` directly means anomaly.  
For RoadAnomaly (older split): pixel value `2` means anomaly → must remap `2 → 1`.

---

### 2.2 Fishyscapes — Reference [2]

**Paper:** Blum et al., *The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation*, 2019  
**Link:** https://arxiv.org/abs/1904.03215

Fishyscapes provides two complementary splits for evaluating OoD detection in
Cityscapes-style road scenes.

#### Fishyscapes LostAndFound (FS L&F)
- **Content:** Based on the LostAndFound dataset — real road scenes with small
  dangerous objects (children toys, shopping bags, boxes) on the road
- **Images:** Real photographs, challenging lighting conditions
- **Labels:**
  - `0` = normal (road surface and standard road scene)
  - `1` = anomaly (the dangerous obstacle)
  - `255` = void/ignore (unlabeled or ambiguous pixels, excluded from evaluation)
- **Challenge:** Anomalies are extremely small relative to image size; severe class
  imbalance (anomaly pixels may be < 0.1% of total pixels)

#### Fishyscapes Static (FS Static)
- **Content:** Synthetic dataset — anomalous objects cut from COCO are pasted onto
  Cityscapes validation images using Poisson image blending
- **Images:** `.jpg` format
- **Labels:** Same format as FS L&F (0/1/255)
- **Challenge:** More controlled than FS L&F; good for benchmarking
- **Note:** Since images are based on Cityscapes, a model trained on Cityscapes
  has a domain advantage on the background — only the pasted objects are anomalous

#### Label Structure (Both Fishyscapes Splits)

```python
# Standard remapping used in evaluation:
ood_gts = 0    # in-distribution (normal pixels)
ood_gts = 1    # out-of-distribution (anomaly pixels)
ood_gts = 255  # ignore (excluded from metric computation)
```

---

### 2.3 Summary Table

| Dataset | Type | Anomalies | Image Ext | GT Values | Key Challenge |
|---|---|---|---|---|---|
| RoadAnomaly | Real | Animals, debris | `.jpg` | 0/2 (2=OoD) | Diversity |
| RoadAnomaly21 | Real | Animals, obstacles | `.png` | 0/1 | Diversity + scale |
| RoadObsticle21 | Real | Boxes, logs, bags | `.webp` | 0/1 | Small size |
| FS LostFound | Real | Small obstacles | `.png` | 0/1/255 | Imbalance |
| FS Static | Synthetic | Pasted COCO objects | `.jpg` | 0/1/255 | Blending artifacts |

---

## 3. Post-Hoc Anomaly Scoring Methods

All three methods below are applied **after** the model's forward pass.
They extract an anomaly signal from the model's existing output without
any retraining.

### ⚠️ Critical Note: ERFNet Returns Raw Logits

ERFNet's decoder ends with a transposed convolution and returns directly:

```python
# From erfnet.py — decoder forward():
output = self.output_conv(output)
return output   # raw logits — NOT softmax probabilities
```

This means `result` from the model has **unbounded values** (positive and negative),
not probabilities in [0, 1]. This distinction is critical:

- Methods using softmax (MSP, MaxEntropy) **must apply softmax first**
- MaxLogit works directly on raw logits

---

### 3.1 MSP — Maximum Softmax Probability

**Paper:** Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and OOD Examples*, 2017

**Formula:**

$$\text{score}_\text{MSP}(\mathbf{x}) = 1 - \max_c \left( \text{softmax}(f(\mathbf{x}))_c \right)$$

where $f(\mathbf{x}) \in \mathbb{R}^C$ are the raw logits and $C$ is the number of classes.

**Intuition:**

When the model is confident about a known class, one softmax probability
dominates (e.g., road = 0.95). The maximum is high → score is low → not anomalous.

When the model sees something unknown, probability spreads across many classes
(e.g., 0.30, 0.25, 0.22, ...). The maximum drops → score rises → anomalous.

```
Known pixel:   softmax → [0.92, 0.04, 0.02, ...]  → max=0.92 → score=0.08
Unknown pixel: softmax → [0.28, 0.24, 0.21, ...]  → max=0.28 → score=0.72
```

**Correct implementation:**

```python
def compute_msp(logits: torch.Tensor) -> np.ndarray:
    # logits: [1, C, H, W] — raw model output
    # Step 1: MUST apply softmax first (ERFNet outputs raw logits)
    probs = torch.softmax(logits, dim=1)       # [1, C, H, W] in [0, 1]
    # Step 2: Take max probability per pixel across classes
    max_prob = probs.max(dim=1).values         # [1, H, W]
    # Step 3: Anomaly score = 1 - confidence
    anomaly_map = 1.0 - max_prob.squeeze(0)   # [H, W]
    return anomaly_map.cpu().numpy()
```

> **Common mistake:** Computing `1 - max(logits)` without softmax.
> Since logits are unbounded, this produces uncalibrated, invalid anomaly scores.

**Advantages:** Simple, fast, works with any softmax classifier  
**Disadvantages:** Softmax overconfidence — even OoD inputs can produce high max prob

---

### 3.2 MaxLogit

**Paper:** Hendrycks et al., *Scaling Out-of-Distribution Detection for Real-World Settings*, ICML 2022  
**arXiv:** https://arxiv.org/abs/1911.11132

**Formula:**

$$\text{score}_\text{MaxLogit}(\mathbf{x}) = -\max_c \left( f(\mathbf{x})_c \right)$$

where $f(\mathbf{x})_c$ are the **raw logits before softmax**.

**Why MaxLogit Often Beats MSP:**

Softmax compresses logit magnitude. Consider two pixels:

| Pixel | Logits | Softmax max |
|---|---|---|
| Known (confident) | [12.0, 1.2, 0.8] | 0.9999 |
| Known (less confident) | [4.0, 1.2, 0.8] | 0.9526 |
| Unknown (OoD) | [3.5, 3.1, 2.9] | 0.4030 |

After softmax, the gap between "confident known" and "less confident known"
nearly disappears. The raw logit magnitude [12.0 vs 4.0 vs 3.5] is much more
discriminative.

**Implementation:**

```python
def compute_maxlogit(logits: torch.Tensor) -> np.ndarray:
    # logits: [1, C, H, W] — raw model output
    # No softmax needed — use raw logits directly
    max_logit = logits.max(dim=1).values          # [1, H, W]
    # Negate: high max logit = confident = not anomalous
    # So we negate to make high score = more anomalous
    anomaly_map = -max_logit.squeeze(0)           # [H, W]
    return anomaly_map.cpu().numpy()
```

**Advantages:** Preserves logit magnitude; empirically stronger than MSP at scale  
**Disadvantages:** Logit scale is not normalized across datasets/models

---

### 3.3 Max Entropy

**Formula (Shannon Entropy):**

$$H(\mathbf{x}) = -\sum_{c=1}^{C} p_c \log p_c$$

$$\text{score}_\text{MaxEntropy}(\mathbf{x}) = H(\mathbf{x})$$

where $p_c = \text{softmax}(f(\mathbf{x}))_c$.

**Properties of Entropy:**

- **Minimum** $H = 0$: all probability on one class → model is certain → not anomalous
- **Maximum** $H = \log C$: uniform distribution over $C$ classes → model is maximally
  uncertain → likely anomalous

For $C = 20$ classes (ERFNet): $H_\text{max} = \log 20 \approx 2.996$

```
Known pixel:   p = [0.92, 0.04, 0.02, ...]  → H ≈ 0.38 (low)
Unknown pixel: p = [0.10, 0.09, 0.08, ...]  → H ≈ 2.89 (high)
```

**Numerical stability:** $\log(0)$ is undefined ($-\infty$). When a class
probability is exactly 0, we add a small epsilon before taking the log:

$$H = -\sum_c p_c \log(p_c + \varepsilon), \quad \varepsilon = 10^{-10}$$

**Implementation:**

```python
def compute_max_entropy(logits: torch.Tensor) -> np.ndarray:
    # logits: [1, C, H, W] — raw model output
    # Step 1: Convert to probabilities
    probs = torch.softmax(logits, dim=1)                    # [1, C, H, W]
    # Step 2: Compute entropy with epsilon for numerical stability
    log_probs = torch.log(probs + 1e-10)                   # [1, C, H, W]
    entropy = -(probs * log_probs).sum(dim=1)              # [1, H, W]
    # Higher entropy = more uncertain = more anomalous
    anomaly_map = entropy.squeeze(0)                        # [H, W]
    return anomaly_map.cpu().numpy()
```

**Optional normalization** to [0, 1]:
```python
import math
entropy_normalized = entropy / math.log(NUM_CLASSES)
```

**Advantages:** Uses the full probability distribution, not just the maximum  
**Disadvantages:** Still subject to softmax overconfidence

---

### 3.4 Comparison of All Three Methods

| Property | MSP | MaxLogit | MaxEntropy |
|---|---|---|---|
| **Formula** | $1 - \max(\text{softmax})$ | $-\max(\text{logits})$ | $-\sum p \log p$ |
| **Applies softmax** | ✅ Yes | ❌ No | ✅ Yes |
| **Uses full distribution** | ❌ No (only max) | ❌ No (only max) | ✅ Yes |
| **Preserves logit scale** | ❌ No | ✅ Yes | ❌ No |
| **Bounded output** | ✅ [0, 1] | ❌ Unbounded | ✅ [0, log C] |
| **Computational cost** | Low | Lowest | Low |
| **Empirical ranking** | Weakest | Often strongest | Between |

---

## 4. Evaluation Metrics

Anomaly segmentation evaluation is done at the **pixel level**.
Each pixel is classified as anomalous (1) or normal (0) based on the anomaly score.
Since the anomaly score is continuous, we sweep over all possible thresholds.

### 4.1 AUPR — Area Under Precision-Recall Curve (= AUPRC)

**Higher is better. Range: [0, 1].**

For each threshold $\tau$:
- **Precision** = $\frac{TP}{TP + FP}$ — of predicted anomalies, how many are real?
- **Recall (TPR)** = $\frac{TP}{TP + FN}$ — of real anomalies, how many detected?

AUPR integrates precision over all recall values:

$$\text{AUPR} = \int_0^1 P(R)\, dR$$

Preferred over AUROC for **imbalanced datasets** (like anomaly segmentation,
where anomaly pixels are rare). Computed in this project via:

```python
from sklearn.metrics import average_precision_score
aupr = average_precision_score(y_true, anomaly_scores)
```

### 4.2 FPR@95TPR — False Positive Rate at 95% True Positive Rate

**Lower is better. Range: [0, 1].**

Fix the threshold at the point where **95% of anomaly pixels are detected** (TPR = 0.95).
At that threshold, measure the fraction of normal pixels incorrectly flagged:

$$\text{FPR@95} = \frac{FP}{FP + TN} \bigg|_{\text{TPR} = 0.95}$$

**Why this metric matters for autonomous driving:**

A self-driving system must detect almost all real road anomalies (high TPR).
The question is: at that detection sensitivity, how many false alarms are generated?
False alarms cause unnecessary braking or steering — a safety and comfort issue.

```python
from ood_metrics import fpr_at_95_tpr
fpr95 = fpr_at_95_tpr(anomaly_scores, y_true)
```

### 4.3 Summary

| Metric | Direction | Meaning |
|---|---|---|
| AUPR | ↑ Higher | Better overall anomaly ranking across thresholds |
| FPR@95 | ↓ Lower | Fewer false alarms when catching 95% of anomalies |

---

## 5. Advanced Methods (Beyond Pixel Baselines)

### 5.1 RbA — Rejected by All (Reference [7])

**Paper:** Nayal et al., *RbA: Segmenting Unknown Regions Rejected by All*, ICCV 2023  
**arXiv:** https://arxiv.org/abs/2211.14293

RbA is a method designed specifically for **mask-based architectures**
(Mask2Former, MaskFormer, EoMT) rather than pixel-based models like ERFNet.

**Core insight:** In mask architectures, each object query specializes in
predicting one class — it behaves like a one-vs-all classifier. Unknown regions
are those that are **rejected by every query** — no query confidently claims them.

**RbA Score:**

For a pixel $p$, let $Q = \{q_1, ..., q_K\}$ be the set of mask queries.
Each query $q_i$ produces:
- A class probability vector $\mathbf{p}_i \in \mathbb{R}^C$
- A mask confidence $m_i(p) \in [0, 1]$

The per-query rejection score for pixel $p$:

$$r_i(p) = 1 - \max_c(p_{i,c}) \cdot m_i(p)$$

The RbA anomaly score aggregates across all queries:

$$\text{score}_\text{RbA}(p) = \prod_{i=1}^{K} r_i(p)$$

High product → rejected by all queries → anomalous.

**Why it works better than pixel-based methods:**
- Produces smoother, object-shaped anomaly regions (not noisy per-pixel)
- Leverages the structure of mask proposals
- No retraining required on OoD data (post-hoc)

> **Note:** RbA requires a mask architecture output. It **cannot** be applied to
> ERFNet (which is pixel-based). It is used with EoMT in Step 8 of this project.

---

### 5.2 MaxLogit at Scale (Reference [8])

**Paper:** Hendrycks et al., *Scaling Out-of-Distribution Detection for Real-World Settings*, ICML 2022  
**arXiv:** https://arxiv.org/abs/1911.11132

This paper is the origin of the **MaxLogit** baseline and demonstrates that:

1. MSP degrades as the number of classes grows (more classes = more probability
   dilution = less discriminative max)
2. MaxLogit is more robust because it operates before the softmax squashing
3. MaxLogit outperforms MSP in large-scale multiclass, multi-label, and
   segmentation settings

The paper also introduces the **CAOS** (Combined Anomalous Object Segmentation)
benchmark and the **StreetHazards** dataset for anomaly segmentation evaluation.

---

## 6. Research Context and Limitations

### Why Post-Hoc Methods Matter

- **No retraining required** — drop-in improvement on any pretrained model
- **Fast to evaluate** — just change the score function
- **Essential baseline** — any stronger method must beat these first

### Limitations of Pixel-Based Baselines

| Limitation | Description |
|---|---|
| Softmax overconfidence | Probabilities always sum to 1; model cannot express "I don't know" |
| No spatial context | Each pixel scored independently; no object-level reasoning |
| No OoD training signal | Model never learned to distinguish known vs unknown |
| Threshold sensitivity | Optimal threshold varies by dataset and method |

### What Modern Methods Do Differently

| Method | Key Idea |
|---|---|
| **RbA** [7] | Mask architecture queries as one-vs-all classifiers |
| **Outlier Exposure** | Train with synthetic OoD examples alongside known classes |
| **Energy-based** | Use energy score $E = -\log \sum_c e^{f_c}$ instead of softmax |
| **Normalizing Flows** | Model density of feature space; flag low-density regions |

---

## 7. References

| # | Citation |
|---|---|
| [1] | Blum et al., *SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation*, NeurIPS 2021. https://arxiv.org/abs/2104.14812 |
| [2] | Blum et al., *The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation*, IJCV 2021. https://arxiv.org/abs/1904.03215 |
| [7] | Nayal et al., *RbA: Segmenting Unknown Regions Rejected by All*, ICCV 2023. https://arxiv.org/abs/2211.14293 |
| [8] | Hendrycks et al., *Scaling Out-of-Distribution Detection for Real-World Settings*, ICML 2022. https://arxiv.org/abs/1911.11132 |
```