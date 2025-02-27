from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List
from src.models.model import DownstreamModel, MAEEncoder, load_vit_weights


class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.r

class LoRAQKVLinear(nn.Linear, LoRALayer):
    """LoRA implemented in Q, K, V attention matrices"""
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.1,
        enable_lora: List[bool] = [True, True, True],  # Enable LoRA for Q, K, V
        **kwargs
    ):
        # First call Linear's init
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # Then call LoRA's init
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

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
        
        # Initialize weights for LoRA
        self.reset_lora_parameters()
        
        # Freeze the original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self):
        """Reset LoRA specific parameters"""
        for lora_A, lora_B, enable in zip(self.lora_A, self.lora_B, self.enable_lora):
            if enable:
                # Initialize weights with kaiming
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                nn.init.zeros_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through the original linear layer
        original_output = F.linear(x, self.weight, self.bias)  # Shape: [B, L, 3*H]
        
        # Early return if no LoRA is enabled
        if not any(self.enable_lora):
            return original_output
        
        # Initialize output with the same shape as original
        B, L, _ = x.shape
        lora_output = torch.zeros_like(original_output)  # Shape: [B, L, 3*H]
        
        # Apply LoRA for Q, K, V separately
        for i in range(3):
            if self.enable_lora[i]:
                # Get the slice corresponding to Q, K, or V
                start_idx = i * self.head_dim
                end_idx = (i + 1) * self.head_dim
                
                # Compute LoRA contribution
                dropped_x = self.lora_dropout(x)
                # Access individual parameters from ParameterList and then transpose
                lora_delta = dropped_x @ torch.transpose(self.lora_A[i], 0, 1) @ torch.transpose(self.lora_B[i], 0, 1)
                lora_delta = lora_delta * self.scaling
                
                # Add to the corresponding slice
                lora_output[:, :, start_idx:end_idx] = lora_delta
        
        return original_output + lora_output
class LoRAProjLinear(nn.Linear, LoRALayer):
    """LoRA implemented in output projection matrix"""
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.1,
        **kwargs
    ):
        # First call Linear's init
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # Then call LoRA's init
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize weights for LoRA
        self.reset_lora_parameters()
        
        # Freeze the original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
    def reset_parameters(self):
        """Reset parameters for base Linear layer"""
        nn.Linear.reset_parameters(self)
        
    def reset_lora_parameters(self):
        """Reset LoRA specific parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through the original linear layer
        original_output = F.linear(x, self.weight, self.bias)
        
        # Compute LoRA contribution
        dropped_x = self.lora_dropout(x)
        lora_delta = (
            dropped_x @ 
            torch.transpose(self.lora_A, 0, 1) @ 
            torch.transpose(self.lora_B, 0, 1)
        ) * self.scaling
        
        return original_output + lora_delta    

class LoRADownstreamModel(DownstreamModel):
    """DownstreamModel with LoRA applied to attention layers"""
    
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_chans=202,
        head_type="conv",
        model_size='base',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    ):
        # Initialize the original model
        super().__init__(img_size, patch_size, in_chans, head_type, model_size)
        
        # Store LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Apply LoRA to attention layers
        self._apply_lora_to_attention(lora_r, lora_alpha, lora_dropout)
        
        # Set up parameter training flags
        self._setup_parameter_requires_grad()
        
    def _apply_lora_to_attention(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to attention layers"""
        vit = self.encoder.vit
        
        # Apply to each transformer block
        for block in vit.blocks:
            # Convert QKV projection
            if hasattr(block.attn, 'qkv'):
                block.attn.qkv = LoRAQKVLinear(
                    block.attn.qkv.in_features,
                    block.attn.qkv.out_features,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=True if block.attn.qkv.bias is not None else False,
                )
            
            # Convert output projection
            if hasattr(block.attn, 'proj'):
                block.attn.proj = LoRAProjLinear(
                    block.attn.proj.in_features,
                    block.attn.proj.out_features,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=True if block.attn.proj.bias is not None else False,
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
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'lora': lora_params,
            'head': head_params
        }

    def get_lora_state(self):
        """Get information about LoRA structure and state"""
        lora_state = {
            'config': {
                'rank': self.lora_r,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout
            },
            'layers': []
        }
        
        # Collect information about each LoRA layer
        for name, module in self.named_modules():
            if isinstance(module, LoRAQKVLinear):
                layer_info = {
                    'name': name,
                    'type': 'QKV',
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
                lora_state['layers'].append(layer_info)
            elif isinstance(module, LoRAProjLinear):
                layer_info = {
                    'name': name,
                    'type': 'Proj',
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
                lora_state['layers'].append(layer_info)
                
        return lora_state
    
    def save_lora_weights(self, path):
        """Save only LoRA weights"""
        lora_state_dict = {
            k: v for k, v in self.state_dict().items()
            if 'lora_' in k
        }
        torch.save(lora_state_dict, path)
        print(f"Saved LoRA weights to {path}")
        
    def merge_lora_weights(self):
        """Merge LoRA weights into base model"""
        for name, module in self.named_modules():
            if isinstance(module, (LoRAQKVLinear, LoRAProjLinear)):
                # Handle merging differently for QKV and Proj layers
                if isinstance(module, LoRAQKVLinear):
                    qkv_weight = module.weight.data.view(3, -1, module.in_features)
                    for i in range(3):
                        if module.enable_lora[i]:
                            # Calculate delta weight with proper transposition
                            delta = (
                                module.lora_B[i] @ module.lora_A[i]  # This gives shape compatible with original weight
                            ) * module.scaling
                            # Ensure delta has the right shape before adding
                            delta = delta.view_as(qkv_weight[i])
                            qkv_weight[i] += delta
                    module.weight.data = qkv_weight.view(-1, module.in_features)
                else:  # LoRAProjLinear
                    # Compute merged weights for projection
                    delta = (
                        module.lora_B @ module.lora_A  # This gives shape compatible with original weight
                    ) * module.scaling
                    # Add to the original weights
                    module.weight.data += delta
                
                # Reset LoRA weights
                with torch.no_grad():
                    if isinstance(module, LoRAQKVLinear):
                        for lora_A, lora_B in zip(module.lora_A, module.lora_B):
                            nn.init.zeros_(lora_A)
                            nn.init.zeros_(lora_B)
                    else:
                        nn.init.zeros_(module.lora_A)
                        nn.init.zeros_(module.lora_B)
        return self
if __name__ == "__main__":
    # Test code for the model
    mae_pth_path = "/home/akram/Downloads/mae.pth"  
    fine_tuned_checkpoint_path = "./mae_finetuning_results/checkpoint_epoch_99.pt"  
    head_type = "unet"  #  "fc", "conv", "residual", or "unet" to try different heads

    try:
        # Create model with LoRA
        model = LoRADownstreamModel(
            img_size=128,
            patch_size=4,
            in_chans=202,
            head_type=head_type,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        # Load pretrained weights
        model = load_vit_weights(model, mae_pth_path)
        print("Loaded pretrained weights successfully")

        # Load fine-tuned weights if they exist
        if Path(fine_tuned_checkpoint_path).exists():
            fine_tuned_checkpoint = torch.load(fine_tuned_checkpoint_path, map_location='cpu')
            model.load_state_dict(fine_tuned_checkpoint['model_state_dict'], strict=True)
            print("Loaded fine-tuned weights successfully")
        else:
            print("No fine-tuned weights found, using only pretrained weights")

        # Print model statistics
        stats = model.get_parameter_groups()
        print("\n=== Parameter Statistics ===")
        print(f"Total parameters: {stats['total']:,}")
        print(f"Frozen parameters: {stats['frozen']:,}")
        print(f"LoRA trainable parameters: {stats['lora']:,}")
        print(f"Head trainable parameters: {stats['head']:,}")
        print(f"Total trainable parameters: {stats['trainable']:,}")
        
        # Print attention layer information
        print("\n=== Attention Layer Structure ===")
        qkv_count = 0
        proj_count = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRAQKVLinear):
                qkv_count += 1
                print(f"\nQKV Layer {qkv_count}:")
                print(f"  Input features: {module.in_features}")
                print(f"  Output features: {module.out_features}")
            elif isinstance(module, LoRAProjLinear):
                proj_count += 1
                print(f"\nProj Layer {proj_count}:")
                print(f"  Input features: {module.in_features}")
                print(f"  Output features: {module.out_features}")
        
        print(f"\nTotal QKV layers with LoRA: {qkv_count}")
        print(f"Total Proj layers with LoRA: {proj_count}")

    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise