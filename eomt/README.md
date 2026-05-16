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

## Shared classes pipeline

These are the commands to run for the shared classes pipeline. You can find the list of shared classes in the file `shared.py`.

### COCO

To evaluate the COCO trained model on the Cityscapes dataset, run:

```bash
python3 eval_shared_miou.py \
  --config configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --ckpt /path/to/coco_ckpt \
  --cityscapes-path /path/to/cityscapes \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --no-masked-attn-enabled \
  --wandb-mode online
```

### Cityscapes

To evaluate the Cityscapes trained model on the Cityscapes dataset, run:

```bash
python3 eval_shared_miou.py \
  --config configs/dinov2/coco/panoptic/eomt_base_640.yaml \
  --ckpt /path/to/cityscapes_ckpt \
  --cityscapes-path /path/to/cityscapes \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2 \
  --no-masked-attn-enabled \
  --wandb-mode online
```
