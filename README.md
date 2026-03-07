# BiRefNet LoRA Fine-Tuning

This repository provides a clean, modular, and efficient pipeline for fine-tuning the [Bi-directional Reference Network (BiRefNet)](https://github.com/ZhengPeng7/BiRefNet) for image segmentation tasks using **Low-Rank Adaptation (LoRA)**. 

By applying LoRA adapters to both the Swin Transformer backbone and the decoder, the project allows for efficient model adaptation with significantly fewer trainable parameters while keeping the original baseline weights frozen.

## Features

- **Efficient Fine-Tuning**: Integrates LoRA (`LoRALinear` and `LoRAConv2d`) for parameter-efficient transfer learning.
- **Modular Pipeline**: Clean separation of data loading, model building, configuration handling, and training logic.
- **Config-Driven**: Easily configure datasets, training hyperparameters, and model settings via `src/config/finetune.yaml`.
- **Automatic Tracking**: Automatically saves configuration files, dataset splits (`train.csv`, `valid.csv`), and logs metrics to TensorBoard for every run.
- **Robust Augmentations**: Built-in data augmentation pipeline using [Albumentations](https://github.com/albumentations-team/albumentations).

## Directory Structure

```text
.
├── src/
│   ├── config/      # Configuration files (e.g., finetune.yaml)
│   ├── data/        # Data loading and augmentation (Dataset classes)
│   ├── finetune/    # Fine-tuning logic, LoRA adapters, custom Loss, and Trainer
│   ├── models/      # Core BiRefNet architecture (Backbone, ASPP, Decoder)
│   ├── utils/       # Utility functions (I/O, Config dictionaries)
│   └── build.py     # Builder functions for dataloaders, model, and trainer
├── train.py         # Main entry point for training
└── .gitignore       # Git ignore settings (ignores weights, runs, and local data)
```

## Get Started

### 1. Prerequisites

Ensure you have a Python environment set up with PyTorch. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset into an image folder and a mask folder. The filenames (excluding extensions) must match exactly between the image and the mask. Update `src/config/finetune.yaml` to point to your data directories:

```yaml
data:
  img_dir: "data/images"
  mask_dir: "data/masks"
  size: [1024, 1024]
  split_ratio: 0.1 # 10% of data will be used for validation
```

### 3. Pre-trained Weights

Download the pre-trained BiRefNet weights and place them according to the path defined in your configuration file. For example:
- `weight/BiRefNet-general-epoch_244.pth`

### 4. Configuration

Adjust the hyper-parameters in `src/config/finetune.yaml` based on your hardware and requirements. Key settings include:
- `train.steps`: Total training steps
- `train.batch`: Batch size
- `train.lr`: Learning rate
- `lora.rank` & `lora.alpha`: LoRA configuration

### 5. Training

Start the fine-tuning process by simply running:

```bash
python train.py
```

## Output

Every time you run the training script, a new directory is created under `run/<timestamp>/`. It contains:
- `logs/`: TensorBoard logs measuring Training Loss, Segmentation Loss, Aux Loss, and Validation Loss.
- `config.yaml`: A copy of the configuration used for this specific run.
- `train.csv` / `valid.csv`: A record of exactly which files were used in the train and validation splits.
- `weights/`: Checkpoints of the trained LoRA adapters (`last.pth` saved periodically, and `model.pth` saved at the end of training).

## References

- **Original Repository**: [BiRefNet on GitHub](https://github.com/ZhengPeng7/BiRefNet)
- **Paper**: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation (arXiv:2401.03407)](https://arxiv.org/abs/2401.03407)