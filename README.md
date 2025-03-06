# Hyperspectral Image Denoising with Vision Transformers

This project implements hyperspectral image denoising using pretrained Vision Transformers (ViT) with various fine-tuning approaches. The project explores three key methods:

1. **Full Fine-tuning**: Training all model parameters
2. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning with low-rank matrices
3. **QLoRA (Quantized Low-Rank Adaptation)**: Further efficiency with quantized weights

The implementation demonstrates how parameter-efficient methods can achieve comparable or better results than full fine-tuning while significantly reducing the number of trainable parameters.

## Features

- **Multiple Decoder Head Types**: FC, Convolutional, Residual, and UNet
- **Parameter-Efficient Fine-tuning**: LoRA and QLoRA implementations
- **Comprehensive Metrics**: PSNR, MSE, and SSIM for evaluation
- **Visualization Tools**: Training curves and denoising results
- **Memory-Efficient Training**: Gradient accumulation and mixed precision

## Project Structure

```
hyperspectral-denoising/
├── src/                       # Main source code
│   ├── models/                # Neural network models
│   │   ├── model.py           # Base ViT and denoising models
│   │   ├── lora_model.py      # LoRA implementation
│   │   └── qlora_model.py     # QLoRA implementation
│   ├── data/                  # Data handling
│   │   └── dataloader.py      # HySpecNet dataset and loaders
│   ├── utils/                 # Utility functions
│   │   └── utils.py           # Metrics, visualization, etc.
│   └── training/              # Training implementations
│       ├── train.py           # Full fine-tuning
│       ├── train_lora.py      # LoRA training
│       └── train_qlora.py     # QLoRA training
├── docs/                      # Documentation
├── tests/                     # Test scripts
├── run_training.py            # Universal training script
├── requirements.txt           # Dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```


### Dataset Setup

1. Download the HySpecNet-11k dataset from https://hyspecnet.rsim.berlin/
2. Extract the dataset and place it in a directory accessible to your project
3. The dataset should have the following structure:

```
/path/to/hyspecnet-11k/
├── patches/
│   ├── ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z/
│   │   ├── ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X03110438/
│   │   │   ├── ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X03110438-DATA.npy
│   │   │   ├── ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X03110438-QL_PIXELMASK.TIF
│   │   │   └── ...
│   │   ├── ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X04390566/
│   │   │   └── ...
│   │   └── ...
│   ├── ENMAP01-____L2A-DT0000004950_20221103T162443Z_002_V010110_20221118T190246Z/
│   │   └── ...
│   └── ...
├── patch_visuals/
└── splits/
```

4. When running the training script, specify the path to the dataset root directory:

```bash
python run_training.py --method lora --data_path /path/to/hyspecnet-11k --save_dir ./results/lora_r8
```

By default, if not specified, the code looks for the dataset at `/home/akram/dataset_download/hyspecnet-11k`



## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://git.tu-berlin.de/rsim/cv4rs-2024-winter/self-supervised-learning-for-hyperspectral-image-analysis/-/tree/akram
   cd hyperspectral-denoising
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n denoising python=3.10
   conda activate denoising
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Training Models

The `run_training.py` script provides a unified interface for all training methods:

```bash
# Full fine-tuning with UNet head
python run_training.py --method full --save_dir ./results/full_unet --head_type unet

# LoRA with different ranks
python run_training.py --method lora --save_dir ./results/lora_r4 --lora_r 4 --lora_alpha 8
python run_training.py --method lora --save_dir ./results/lora_r16 --lora_r 16 --lora_alpha 32

# QLoRA with different quantization
python run_training.py --method qlora --save_dir ./results/qlora_nf4 --quant_type nf4
```

### Memory Management

For large models or limited GPU memory:

```bash
python run_training.py --method lora --save_dir ./results/lora_r16 \
    --batch_size 4 --gradient_accumulation_steps 4 --empty_cache
```

### Available Parameters

Common parameters:
- `--method`: Training method (`full`, `lora`, or `qlora`)
- `--data_path`: Path to dataset directory
- `--save_dir`: Directory to save results
- `--batch_size`: Batch size
- `--head_type`: Head architecture (`fc`, `conv`, `residual`, or `unet`)

LoRA/QLoRA specific:
- `--lora_r`: Rank for LoRA adaptation
- `--lora_alpha`: Alpha scaling factor
- `--quant_type`: Quantization type for QLoRA (`fp4` or `nf4`)

Run `python run_training.py --help` for a complete list of parameters.

## Evaluation and Visualization

The training scripts automatically:
- Save model checkpoints
- Generate visualizations of results
- Plot training and validation curves
- Record metrics (PSNR, loss)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The ViT implementation is based on the timm library
- The HySpecNet-11k dataset is used for training and evaluation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{reference,
  author = {Akram },
  title = {Self-Supervised Learning for Hyperspectral Image Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {Gitlab repository},
  howpublished = {\url{https://git.tu-berlin.de/rsim/cv4rs-2024-winter/self-supervised-learning-for-hyperspectral-image-analysis/-/tree/akram?ref_type=heads}}
}
```
