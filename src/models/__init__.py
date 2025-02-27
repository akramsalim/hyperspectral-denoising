"""
Models module containing all neural network architectures for the project.

This module includes:

Base Models:
- ModifiedViT: Vision Transformer modified for hyperspectral data
- MAEEncoder: Encoder from the pretrained MAE model
- DownstreamModel: Main downstream model with different head options

Head Implementations:
- FCHead: Fully connected head for reconstruction
- ConvHead: Convolutional head for reconstruction
- ResidualBlockHead: Head with residual connections
- UNetHead: U-Net architecture for detailed reconstruction

LoRA Implementation:
- LoRADownstreamModel: DownstreamModel with Low-Rank Adaptation
- LoRAQKVLinear: LoRA implementation for query, key, value projections
- LoRAProjLinear: LoRA implementation for output projections

QLoRA Implementation:
- QLoRADownstreamModel: DownstreamModel with Quantized Low-Rank Adaptation
- QLoRAQKVLinear: QLoRA implementation for query, key, value projections
- QLoRAProjLinear: QLoRA implementation for output projections

Utility Functions:
- load_vit_weights: Function to load pretrained ViT weights
"""

# Import base model components
from .model import (
    ModifiedViT,
    MAEEncoder,
    DownstreamModel,
    FCHead,
    ConvHead,
    ResidualBlockHead,
    UNetHead,
    load_vit_weights
)

# Import LoRA components
from .lora_model import (
    LoRALayer,
    LoRAQKVLinear,
    LoRAProjLinear,
    LoRADownstreamModel
)

# Import QLoRA components
from .qlora_model import (
    QLoRALayer,
    QLoRAQKVLinear,
    QLoRAProjLinear,
    QLoRADownstreamModel
)

# Define exports
__all__ = [
    # Base model
    'ModifiedViT',
    'MAEEncoder',
    'DownstreamModel',
    'FCHead',
    'ConvHead',
    'ResidualBlockHead',
    'UNetHead',
    'load_vit_weights',
    
    # LoRA implementation
    'LoRALayer',
    'LoRAQKVLinear',
    'LoRAProjLinear',
    'LoRADownstreamModel',
    
    # QLoRA implementation
    'QLoRALayer',
    'QLoRAQKVLinear',
    'QLoRAProjLinear',
    'QLoRADownstreamModel'
]