import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List
from src.models.model import DownstreamModel, MAEEncoder, load_vit_weights
from bitsandbytes.nn import Linear4bit, Params4bit
import bitsandbytes as bnb


class QLoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.r
        
    def merge_weights(self):
        """Merge LoRA weights with the quantized weights"""
        raise NotImplementedError("Each QLoRA layer must implement merge_weights")

class QLoRAQKVLinear(Linear4bit, QLoRALayer):
    """QLoRA implemented in Q, K, V attention matrices using 4-bit quantization"""
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.1,
        enable_lora: List[bool] = [True, True, True],  # Enable LoRA for Q, K, V
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4",  # Can be 'fp4' or 'nf4'
        **kwargs
    ):
        # Initialize 4-bit quantized layer
        Linear4bit.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get('bias', True),
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type
        )
        # Initialize LoRA part
        QLoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.quant_type = quant_type
        
        assert out_features % 3 == 0, "QKV dimension must be divisible by 3"
        self.enable_lora = enable_lora
        self.head_dim = out_features // 3
        
        # Initialize A and B matrices for Q, K, V
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(r, in_features)) if enable 
            else nn.Parameter(torch.zeros(1)) # Dummy tensor when LoRA is disabled
            for enable in enable_lora
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.head_dim, r)) if enable 
            else nn.Parameter(torch.zeros(1)) # Dummy tensor when LoRA is disabled
            for enable in enable_lora
        ])
        
        # Reset LoRA parameters
        self.reset_lora_parameters()
        
        # Freeze the quantized weights
        self.weight.requires_grad = False
        
    def reset_lora_parameters(self):
        """Reset LoRA specific parameters"""
        for lora_A, lora_B, enable in zip(self.lora_A, self.lora_B, self.enable_lora):
            if enable:
                # Initialize weights with kaiming
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                nn.init.zeros_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through the quantized layer
        base_output = super().forward(x)
        
        # Early return if no LoRA is enabled
        if not any(self.enable_lora):
            return base_output
        
        # Initialize output with the same shape as original
        B, L, _ = x.shape
        lora_output = torch.zeros_like(base_output)
        
        # Apply LoRA for Q, K, V separately
        for i in range(3):
            if self.enable_lora[i]:
                start_idx = i * self.head_dim
                end_idx = (i + 1) * self.head_dim
                
                dropped_x = self.lora_dropout(x)
                lora_delta = (dropped_x @ 
                            self.lora_A[i].T @ 
                            self.lora_B[i].T) * self.scaling
                
                lora_output[:, :, start_idx:end_idx] = lora_delta
        
        return base_output + lora_output.to(base_output.dtype)

class QLoRAProjLinear(Linear4bit, QLoRALayer):
    """QLoRA implemented in output projection matrix using 4-bit quantization"""
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.1,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4",
        **kwargs
    ):
        # Initialize 4-bit quantized layer
        Linear4bit.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get('bias', True),
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type
        )
        # Initialize LoRA part
        QLoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.quant_type = quant_type
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Reset LoRA parameters
        self.reset_lora_parameters()
        
        # Freeze the quantized weights
        self.weight.requires_grad = False
            
    def reset_lora_parameters(self):
        """Reset LoRA specific parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through the quantized layer
        base_output = super().forward(x)
        
        # Compute LoRA contribution
        dropped_x = self.lora_dropout(x)
        lora_output = (dropped_x @ 
                      self.lora_A.T @ 
                      self.lora_B.T) * self.scaling
        
        return base_output + lora_output.to(base_output.dtype)

class QLoRADownstreamModel(DownstreamModel):
    """DownstreamModel with QLoRA applied to attention layers"""
    
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_chans=202,
        head_type="conv",
        model_size='base',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        quant_type="nf4",  # Can be 'fp4' or 'nf4'
        compute_dtype=torch.float16
    ):
        # Initialize the original model
        super().__init__(img_size, patch_size, in_chans, head_type, model_size)
        
        # Store LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        
        # Apply QLoRA to attention layers
        self._apply_qlora_to_attention(lora_r, lora_alpha, lora_dropout)
        
        # Set up parameter training flags
        self._setup_parameter_requires_grad()
        
    def _apply_qlora_to_attention(self, r: int, alpha: int, dropout: float):
        """Apply QLoRA to attention layers"""
        vit = self.encoder.vit
        
        # Apply to each transformer block
        for block in vit.blocks:
            # Convert QKV projection
            if hasattr(block.attn, 'qkv'):
                block.attn.qkv = QLoRAQKVLinear(
                    block.attn.qkv.in_features,
                    block.attn.qkv.out_features,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=True if block.attn.qkv.bias is not None else False,
                    quant_type=self.quant_type,
                    compute_dtype=self.compute_dtype
                )
            
            # Convert output projection
            if hasattr(block.attn, 'proj'):
                block.attn.proj = QLoRAProjLinear(
                    block.attn.proj.in_features,
                    block.attn.proj.out_features,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=True if block.attn.proj.bias is not None else False,
                    quant_type=self.quant_type,
                    compute_dtype=self.compute_dtype
                )
    
    def _setup_parameter_requires_grad(self):
        """Set up which parameters should be trained"""
        # Freeze all encoder parameters except LoRA
        for name, param in self.encoder.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Keep head parameters trainable
        for param in self.head.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self):
        """Count different types of parameters in the model"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        lora_params = 0
        head_params = 0
        quantized_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if 'lora_' in name:
                    lora_params += num_params
                elif 'head.' in name:
                    head_params += num_params
            else:
                frozen_params += num_params
                if isinstance(param, Params4bit):
                    quantized_params += num_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'lora': lora_params,
            'head': head_params,
            'quantized': quantized_params
        }

    def get_qlora_state(self):
        """Get information about QLoRA structure and state"""
        qlora_state = {
            'config': {
                'rank': self.lora_r,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout,
                'quant_type': self.quant_type
            },
            'layers': []
        }
        
        # Collect information about each QLoRA layer
        for name, module in self.named_modules():
            if isinstance(module, (QLoRAQKVLinear, QLoRAProjLinear)):
                layer_info = {
                    'name': name,
                    'type': 'QKV' if isinstance(module, QLoRAQKVLinear) else 'Proj',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'quant_type': module.quant_type
                }
                qlora_state['layers'].append(layer_info)
                
        return qlora_state
    
    def save_qlora_weights(self, path):
        """Save only QLoRA weights"""
        qlora_state_dict = {
            k: v for k, v in self.state_dict().items()
            if 'lora_' in k
        }
        torch.save(qlora_state_dict, path)
        print(f"Saved QLoRA weights to {path}")

if __name__ == "__main__":
    # Test code for the model
    mae_pth_path = "/path/to/mae.pth"
    
    try:
        # Create model with QLoRA
        model = QLoRADownstreamModel(
            img_size=128,
            patch_size=4,
            in_chans=202,
            head_type="unet",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            quant_type="nf4"
        )
        
        # Load pretrained weights
        model = load_vit_weights(model, mae_pth_path)
        print("Loaded pretrained weights successfully")
        
        # Print model statistics
        stats = model.get_parameter_groups()
        print("\n=== Parameter Statistics ===")
        print(f"Total parameters: {stats['total']:,}")
        print(f"Frozen parameters: {stats['frozen']:,}")
        print(f"Quantized parameters: {stats['quantized']:,}")
        print(f"LoRA trainable parameters: {stats['lora']:,}")
        print(f"Head trainable parameters: {stats['head']:,}")
        print(f"Total trainable parameters: {stats['trainable']:,}")
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise
