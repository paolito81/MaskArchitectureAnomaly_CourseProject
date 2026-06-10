# Mask Architecture Anomaly — Comprehensive Road-Scene Understanding

**Course project, Politecnico di Torino — Group 22**

Hossein Ahmadzadeh · Omar Wafaay · Paolo Pusterla · Valeria Intini

Joint semantic and anomaly segmentation for road scenes, comparing
**ERFNet** (pixel-based) and **EoMT** (mask-classification) under a
shared 7-class evaluation, a LoRA fine-tune of the COCO-pretrained
EoMT, and four post-hoc anomaly scoring methods
(MSP / MaxLogit / MaxEntropy / RbA).

Fine-tuned checkpoint and a summary of results are also available
on our GitHub release page.

---

## 1. Requirements

- Python 3.10+
- CUDA-enabled GPU (the fine-tune was run on a single 6–8 GB laptop GPU)
- PyTorch 2.x + CUDA build matching your driver
- PyTorch Lightning 2.x
- Other Python packages:
  `wandb`, `numpy`, `pandas`, `scikit-learn`, `pillow`, `matplotlib`,
  `pyyaml`, `tqdm`, `ood_metrics`

### Conda environment (recommended)

```bash
conda create -n roadscenes python=3.10 -y
conda activate roadscenes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If `requirements.txt` is missing the OOD metrics package, install it
explicitly:

```bash
pip install ood-metrics
```

---

## 2. Repository layout

```
MaskArchitectureAnomaly_CourseProject/
├── eomt/                          # EoMT codebase + LoRA wrappers
│   ├── configs/                   # YAML configs (Cityscapes, COCO, fine-tune)
│   ├── models/                    # eomt.py, vit.py, lora.py
│   ├── training/                  # Lightning module
│   ├── eval_shared_miou.py        # Task 4 / Task 5 shared 7-class eval
│   └── main.py                    # Train / validate entry point
├── eval/                          # Anomaly evaluation pipeline (Task 7 / 8)
│   ├── evalAnomaly.py             # Task 7 — ERFNet pixel-based baselines
│   ├── evalAnomaly_mask_methods.py  # Task 8 — EoMT mask-based methods
│   ├── mask_posthoc.py            # MSP / MaxLogit / MaxEntropy / RbA score functions
│   ├── temperature_sweep.py       # Offline temperature sweep
│   ├── logits_cache.py            # Per-image cache for mask + class logits
│   ├── eomt_adapter.py            # Config-driven EoMT loader
│   └── gt_utils.py                # GT mask loading / label unification
├── erfnet_pytorch/                # ERFNet checkpoint + helper code
├── report/                        # Final report sources
└── README.md
```

---

## 3. Data setup

Set `D:\Project\data` (or your equivalent) as the root and put the
following inside it:

| Folder / file                            | Purpose                                 |
|------------------------------------------|-----------------------------------------|
| `cityscapes/leftImg8bit/`                | Cityscapes images (train + val)         |
| `cityscapes/gtFine/`                     | Cityscapes ground truth                 |
| `eomt_cityscapes.bin`                    | EoMT Cityscapes semantic checkpoint     |
| `eomt_coco.bin`                          | EoMT COCO panoptic checkpoint           |
| `RoadAnomaly/`                           | RoadAnomaly evaluation set              |
| `dataset_AnomalyTrack/`                  | SMIYC AnomalyTrack (RA-21)              |
| `dataset_ObstacleTrack/`                 | SMIYC ObstacleTrack (RO-21)             |
| `fs_lost_found/`                         | Fishyscapes Lost & Found                |
| `fs_static/`                             | Fishyscapes Static                      |

ERFNet weights live under `erfnet_pytorch/save/erfnet_pretrained.pth`.
Our fine-tuned EoMT checkpoint can be downloaded from the GitHub
release; place it at
`D:\Project\data\checkpoints_my_run\eomt_lora_finetuned.ckpt`.

---

## 4. Task 4 — Shared 7-class checkpoint comparison

Evaluate each EoMT checkpoint against Cityscapes-val under the shared
7-class label space (person, car, truck, bus, motorcycle, bicycle,
traffic light).

### EoMT (COCO panoptic) → 78.79 % shared-class mIoU

```powershell
python eval_shared_miou.py `
  --config configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml `
  --ckpt D:\Project\data\eomt_coco.bin `
  --cityscapes-path D:\Project\data `
  --device cuda:0 --batch-size 1 --num-workers 2 `
  --no-masked-attn-enabled `
  --wandb-mode online --wandb-project task4_compare --wandb-name eomt_coco
```

### EoMT (Cityscapes semantic) → 92.25 % shared-class mIoU

```powershell
python eval_shared_miou.py `
  --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml `
  --ckpt D:\Project\data\eomt_cityscapes.bin `
  --cityscapes-path D:\Project\data `
  --device cuda:0 --batch-size 1 --num-workers 2 `
  --no-masked-attn-enabled `
  --wandb-mode online --wandb-project task4_compare --wandb-name eomt_cityscapes
```

Both runs print per-class IoU, the shared mIoU, and the normalised
confusion matrix.

---

## 5. Task 5 — LoRA fine-tune of EoMT-COCO on Cityscapes

### 5.1 Train

LoRA rank 32, scaling α = 64, applied to attention QKV and the two MLP
projections. DINOv2 backbone and the COCO query embeddings stay
frozen; only LoRA adapters + class head + mask head + upsampling
module are trained.

```powershell
python main.py fit `
  -c configs/dinov2/cityscapes/semantic/eomt_base_512_lora_from_coco_my_run.yaml `
  --trainer.devices 1 `
  --data.path D:\Project\data `
  --model.ckpt_path D:\Project\data\eomt_coco.bin `
  --model.load_ckpt_class_head False `
  --trainer.default_root_dir D:\Project\data\checkpoints_my_run `
  --trainer.logger.init_args.project task5_my_run `
  --trainer.logger.init_args.name eomt_lora_my_run `
  --compile_disabled
```

Trains at 512×512, FP16, batch size 1 with gradient accumulation = 8.
Early-stopping monitors `metrics/val_iou_all` with patience 6.
The best checkpoint is saved under
`task5_my_run/<run_id>/checkpoints/best-epoch=*.ckpt`.

### 5.2 Validate — shared 7-class mIoU (matches Task 4 protocol)

```powershell
python eval_shared_miou.py `
  --config configs/dinov2/cityscapes/semantic/eomt_base_512_lora_from_coco_my_run.yaml `
  --ckpt "<path_to_best_ckpt>.ckpt" `
  --cityscapes-path D:\Project\data `
  --device cuda:0 --batch-size 1 --num-workers 2 `
  --no-masked-attn-enabled --src-label-space cityscapes `
  --wandb-mode online --wandb-project task5_my_run --wandb-name eomt_lora_shared7
```

Result: **88.16 % shared-class mIoU** (vs 78.79 % for EoMT-COCO).

### 5.3 Validate — full 19-class Cityscapes mIoU (sanity check)

```powershell
python main.py validate `
  -c configs/dinov2/cityscapes/semantic/eomt_base_512_lora_from_coco_my_run.yaml `
  --trainer.devices 1 --data.batch_size 1 `
  --data.path D:\Project\data `
  --model.ckpt_path "<path_to_best_ckpt>.ckpt" `
  --model.network.init_args.masked_attn_enabled False `
  --trainer.logger.init_args.project task5_my_run `
  --trainer.logger.init_args.name eomt_lora_19class
```

Result: 75.6 % 19-class mIoU.

---

## 6. Task 7 — Pixel-based anomaly baselines (ERFNet)

Computes MSP, MaxLogit, MaxEntropy on the public ERFNet checkpoint
across the five anomaly datasets.

```powershell
cd eval
python evalAnomaly.py `
  --loadDir ../erfnet_pytorch/save/ `
  --loadWeights erfnet_pretrained.pth `
  --loadModel erfnet.py `
  --datadir D:\Project\data
```

Reported metrics: AuPRC and FPR@95 per (method × dataset). Output
example:

```
MSP        | RoadAnomaly        | AUPRC: 12.42% | FPR@95: 82.58%
MAXLOGIT   | RoadAnomaly        | AUPRC: 15.58% | FPR@95: 73.26%
MAXENTROPY | RoadAnomaly        | AUPRC: 12.67% | FPR@95: 82.75%
MSP        | RoadAnomaly21      | AUPRC: 29.09% | FPR@95: 62.56%
... (see report Table 2 for the full set)
```

---

## 7. Task 8 — Mask-based anomaly methods (EoMT)

Three EoMT checkpoints (COCO, Cityscapes, fine-tuned) × four post-hoc
methods (MSP, MaxLogit, MaxEntropy, RbA) × five datasets, plus an
offline MSP temperature sweep.

The pipeline runs in three modes:

1. `cache`  — single forward pass per image, save mask + class logits
2. `score`  — apply post-hoc methods to the cached logits
3. `sweep`  — sweep MSP over 10 temperatures using the same cache

### 7.1 Cache logits for one checkpoint

```powershell
cd eval
python evalAnomaly_mask_methods.py `
  --mode cache `
  --ckpt-tag finetuned_v2 `
  --eomt-config ../eomt/configs/dinov2/cityscapes/semantic/eomt_base_512_lora_from_coco_my_run.yaml `
  --eomt-ckpt "<path_to_best_ckpt>.ckpt" `
  --datadir D:\Project\data `
  --cache-root D:\Project\data\logits_cache `
  --datasets RoadAnomaly RoadAnomaly21 RoadObsticle21 FS_LostFound_full fs_static
```

Repeat with `--ckpt-tag cityscapes` and `--ckpt-tag coco` using the
corresponding configs and `.bin` files.

### 7.2 Score all post-hoc methods (T = 1)

```powershell
python evalAnomaly_mask_methods.py `
  --mode score `
  --ckpt-tag finetuned_v2 `
  --cache-root D:\Project\data\logits_cache `
  --datadir D:\Project\data `
  --methods MSP MAXLOGIT MAXENTROPY RBA
```

Output example:

```
finetuned_v2 | MSP        | RoadAnomaly21      | AUPRC: 76.89% | FPR@95: 10.49%
finetuned_v2 | MAXLOGIT   | RoadAnomaly21      | AUPRC: 67.09% | FPR@95: 20.87%
finetuned_v2 | MAXENTROPY | RoadAnomaly21      | AUPRC: 81.61% | FPR@95:  8.39%
finetuned_v2 | RBA        | RoadAnomaly21      | AUPRC: 47.83% | FPR@95: 97.92%
... (see report Table 2 for the full grid)
```

### 7.3 Temperature sweep (MSP only, on the fine-tuned checkpoint)

```powershell
python evalAnomaly_mask_methods.py `
  --mode sweep `
  --ckpt-tag finetuned_v2 `
  --cache-root D:\Project\data\logits_cache `
  --datadir D:\Project\data `
  --temperatures 0.5 0.75 1.0 1.1 1.5 2.0 3.0 5.0 7.5 10.0
```

The script prints one block per dataset and the AuPRC-best T. Best-T
example:

```
finetuned_v2 | best T = 1.5 | RoadAnomaly21
finetuned_v2 | best T = 2.0 | RoadObsticle21
finetuned_v2 | best T = 3.0 | FS_LostFound_full
finetuned_v2 | best T = 3.0 | fs_static
finetuned_v2 | best T = 2.0 | RoadAnomaly
```

---

## 8. Reproducing the report numbers

After running Task 4, Task 5, Task 7, and Task 8 with the commands
above, the printed values match the report exactly:

- Task 4: EoMT-COCO 78.79 %, EoMT-City 92.25 % shared-class mIoU
- Task 5: EoMT-FT 88.16 % shared-class mIoU (75.6 % 19-class)
- Task 7: ERFNet MaxLogit best on all five datasets (see report
  Table 2, ERFNet block)
- Task 8: EoMT-FT + RbA on FS Static reaches 79.06 % AuPRC at 8.86 %
  FPR@95; EoMT-City + MaxLogit on SMIYC RO-21 reaches 89.24 % AuPRC
  at 0.37 % FPR@95
- Temperature sweep: SMIYC RO-21 MSP improves from 33.91 % to 65.39 %
  AuPRC at T = 2.0

All numbers are reproducible from the cached logits, so the offline
sweep does not require GPU access after the initial caching pass.

---

## 9. Determinism / reproducibility

For bit-level reproducibility of anomaly numbers:

```python
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These flags are set inside the evaluation scripts; they only affect
inference, not the original fine-tune.

---

## 10. License and acknowledgments

The EoMT codebase is reused under its original license (Kerssies et
al., CVPR 2025). LoRA wrappers follow Hu et al., 2021. Cityscapes,
SMIYC, Fishyscapes, and RoadAnomaly datasets are distributed under
their respective licenses; we do not redistribute them in this repo.

For the academic report and the GenAI declarations, see the report for the project.
