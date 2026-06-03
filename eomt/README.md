# EoMT

This is almost the original repository of the authors of EoMT if something is not clear refer to the [original repo](https://github.com/tue-mps/eomt). You will have to use the code in this folder and adapt it with the eval folder to be able to evaluate and train a EoMT model if needed. You can find a EoMT model trained on Cityscapes dataset with the [config file](eomt/configs/dinov2/cityscapes/semantic) at this [link](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Requirements Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a clean Python 3.11 environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.11
conda activate eomt
python3 -m pip install -r requirements.txt
```

This project is not currently set up for Python 3.12+.
Using Python 3.12/3.13 can cause pip to resolve newer PyTorch builds than the ones pinned here, which then conflicts with the rest of the stack.

If you are running in Colab or another shared environment, prefer a fresh virtual environment or uninstall conflicting preinstalled packages before installing the requirements:

```bash
python3 -m pip uninstall -y torch torchao torchvision transformers numpy
python3 -m pip install -r requirements.txt
```

The most common symptoms of an incompatible environment are:
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2`
- `AttributeError: type object 'torch._C.Tag' has no attribute 'needs_fixed_stride_order'`

Those usually mean the environment has drifted to `numpy>=2`, a newer `torch`, or a `torchao` build that does not match the installed PyTorch version.

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation for training

You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch (don't do it, it will be impossible to do it in Colab due to resource contraints):

```bash
python3 main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on Citiscapes segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.

✅ Make sure the total batch size is `devices × batch_size = 16`
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

To fine-tune with LoRA adapters instead of updating the ViT backbone weights, use the LoRA config:

```bash
python3 main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640_lora.yaml \
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False \
  --trainer.devices 1 \
  --data.batch_size 1 \
  --data.path /path/to/dataset
```

The LoRA config keeps the original EoMT fine-tuning flow but injects trainable low-rank adapters into the ViT attention `qkv` projections. The segmentation heads and upscale module remain trainable, while the base ViT weights stay frozen.

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

## Shared classes evaluation

Use `eval_shared_miou.py` to evaluate an EoMT checkpoint on the Cityscapes validation set after remapping model outputs and Cityscapes targets into a common label space. The mappings are defined in `shared_eval/shared.py`.

The default evaluation label space is `shared`, which contains these 7 classes:

```text
person, car, truck, bus, motorcycle, bicycle, traffic light
```

The script also supports `--eval-label-space cityscapes`, which evaluates in the 19-class Cityscapes space. When the source checkpoint is COCO-trained, only Cityscapes classes with a COCO mapping are scored; the unmatched Cityscapes GT classes are ignored.

### Common arguments

- `--config`: source model config used to rebuild the model architecture.
- `--ckpt`: checkpoint to load.
- `--cityscapes-path`: directory containing the Cityscapes zip files.
- `--src-label-space`: source model output space. Defaults to `auto`, inferred from `data.class_path` in the config. Use `coco` or `cityscapes` only if the config cannot be inferred correctly.
- `--eval-label-space`: metric label space. Use `shared` for the 7 shared classes or `cityscapes` for the Cityscapes-space evaluation.
- `--masked-attn-enabled` / `--no-masked-attn-enabled`: inference-time masked-attention override. The default is disabled, so the explicit `--no-masked-attn-enabled` flag is optional.
- `--limit`: optional number of validation images to process for debugging.
- `--wandb-mode`: `disabled`, `offline`, or `online`. Defaults to `disabled`.
- `--wandb-project` and `--wandb-name`: optional W&B overrides. If omitted, the script uses the logger project/name from the config when available.

The script prints per-class IoU, mIoU, valid/ignored pixel counts, a confusion matrix, and a row-normalized confusion matrix. With W&B enabled, it logs the metrics, audit counts, source code, and one qualitative image/GT/prediction example.

### COCO checkpoint on Cityscapes, shared classes

```bash
python3 eval_shared_miou.py \
  --config configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --ckpt /path/to/coco_ckpt \
  --cityscapes-path /path/to/cityscapes \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --eval-label-space shared \
  --wandb-mode online
```

### Cityscapes checkpoint on Cityscapes, shared classes

```bash
python3 eval_shared_miou.py \
  --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --ckpt /path/to/cityscapes_ckpt \
  --cityscapes-path /path/to/cityscapes \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --eval-label-space shared \
  --wandb-mode online
```

### Cityscapes-space evaluation

Use this when you want metrics in the Cityscapes label space instead of only the 7 shared classes.

```bash
python3 eval_shared_miou.py \
  --config /path/to/source_config.yaml \
  --ckpt /path/to/checkpoint \
  --cityscapes-path /path/to/cityscapes \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --eval-label-space cityscapes \
  --wandb-mode online
```

For quick checks, add `--limit 10` and keep `--wandb-mode disabled`.
