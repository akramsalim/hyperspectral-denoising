"""
Hyperspectral Image Denoising with Vision Transformers and Low-Rank Adaptation

This package implements methods for hyperspectral image denoising using
pretrained Vision Transformers (ViT) with various fine-tuning approaches:

1. Full fine-tuning
2. Low-Rank Adaptation (LoRA)
3. Quantized Low-Rank Adaptation (QLoRA)

The project demonstrates how parameter-efficient fine-tuning methods
can achieve comparable results for full fine-tuning while
requiring significantly fewer trainable parameters.

Main components:
- models: Neural network architectures including ViT-based models with
          LoRA and QLoRA implementations
- data: Dataset and dataloader utilities for the HySpecNet dataset
- utils: Common utilities for metrics, visualization, and training
- training: Training loops for different fine-tuning approaches

"""

__version__ = '1.0.0'

# Import key components for easier access
from src.models import (
    DownstreamModel,
    LoRADownstreamModel,
    QLoRADownstreamModel,
    load_vit_weights
)

from src.data import (
    create_dataloaders,
    HySpecNetDataset
)

from src.utils import (
    calculate_psnr,
    visualize_results,
    calculate_metrics,
    EarlyStopping
)

__all__ = [
    # Models
    'DownstreamModel',
    'LoRADownstreamModel',
    'QLoRADownstreamModel',
    'load_vit_weights',
    
    # Data
    'create_dataloaders',
    'HySpecNetDataset',
    
    # Utils
    'calculate_psnr',
    'visualize_results',
    'calculate_metrics',
    'EarlyStopping'
]