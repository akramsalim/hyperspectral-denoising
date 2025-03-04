# Technical Documentation: Hyperspectral Image Denoising with Parameter-Efficient Fine-Tuning

## 1. Project Overview

### 1.1 Purpose and Objectives

This project implements and evaluates parameter-efficient methods for hyperspectral image denoising using pre-trained Vision Transformers (ViT) of Masked Autoencoder. The primary goal is to compare the performance of three fine-tuning approaches with the baseline methods:

1. **Full Fine-tuning**: Training all parameters of the pre-trained model
2. **Low-Rank Adaptation (LoRA)**: Fine-tuning using low-rank decomposition matrices
3. **Quantized Low-Rank Adaptation (QLoRA)**: Combining 4-bit quantization with LoRA

We demonstrate that parameter-efficient methods (LoRA and QLoRA) can achieve comparable performance to full fine-tuning while significantly reducing memory requirements and computational costs.

### 1.2 Challenges in Hyperspectral Image Analysis

Hyperspectral images present several unique challenges:
- High dimensionality (hundreds of spectral bands)
- Increased computational complexity
- Mixed noise types (Gaussian, stripe)
- Memory constraints when training deep learning models

### 1.3 Key Features

- Pre-trained ViT encoder with frozen weights
- Multiple decoder head architectures (FC, Conv, Residual, UNet)
- LoRA adaptation for parameter-efficient fine-tuning
- QLoRA enhancement for further memory reduction
- Comprehensive benchmarking against traditional and deep learning-based methods
- Robust evaluation on both synthetic and real hyperspectral datasets
- Memory-efficient training with gradient accumulation and mixed precision

## 2. Code Architecture

### 2.1 Directory Structure

```
project/
├── src/                       # Source code
│   ├── models/                # Model definitions
│   │   ├── model.py           # Base model and head implementations
│   │   ├── lora_model.py      # LoRA implementation
│   │   └── qlora_model.py     # QLoRA implementation
│   ├── data/                  # Data handling
│   │   └── dataloader.py      # Dataset and data loading utilities
│   ├── utils/                 # Utility functions
│   │   └── utils.py           # Metrics, visualization, and training utilities
│   └── training/              # Training implementations
│       ├── train.py           # Full fine-tuning trainer
│       ├── train_lora.py      # LoRA training
│       └── train_qlora.py     # QLoRA training
├── tests/                     # Test scripts
├── run_training.py            # Universal training script
├── setup.py                   # Package installation script
└── requirements.txt           # Dependencies
```

### 2.2 Architecture Overview

The project follows a modular design with clear separation of concerns:

1. **Models**: Define the neural network architectures
2. **Data**: Handle dataset loading and preprocessing
3. **Utils**: Provide metrics, visualization, and helper functions
4. **Training**: Implement different training strategies
5. **Entry Point**: Unified interface for all training methods

## 3. Model Architecture

### 3.1 Base Model Structure

The base model consists of two main components:

1. **Encoder**: Pre-trained MAE ViT
2. **Decoder**: Various head architectures for reconstruction

```python
class DownstreamModel(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=202, head_type="conv", model_size='base'):
        super().__init__()
        
        # Using MAEEncoder instead of raw VisionTransformer
        self.encoder = MAEEncoder(
            model_size=model_size,
            num_input_channels=in_chans,
            img_size=img_size
        )
        
        # Select head type
        if head_type == "fc":
            self.head = FCHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "conv":
            self.head = ConvHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "residual":
            self.head = ResidualBlockHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "unet":
            self.head = UNetHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Freeze encoder, train head
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.encoder(x)  # [B,1025,384]
        
        features = features[:, 1:, :]  # Remove CLS token [B,1024,384]
        
        B, N, D = features.shape
        h = w = int((self.img_size / self.patch_size))

        if isinstance(self.head, FCHead):
            patches = self.head(features)
            patches = patches.reshape(B, h, w, self.in_chans, self.patch_size, self.patch_size)
            patches = patches.permute(0,3,1,4,2,5)
            out = patches.reshape(B, self.in_chans, h*self.patch_size, w*self.patch_size)
        else:
            # For conv-based heads
            feat_map = features.permute(0,2,1).reshape(B, D, h, w)
            out = self.head(feat_map)
            
        return out
```

### 3.2 Decoder Head Architectures

#### 3.2.1 FCHead

A simple fully-connected head that processes each patch embedding independently:

```python
class FCHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.fc = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def forward(self, features):
        B, N, D = features.shape
        patches = []
        for i in range(N):
            patch_embed = features[:, i, :]
            patch_pixels = self.fc(patch_embed)
            patch_pixels = patch_pixels.reshape(B, self.in_chans, self.patch_size, self.patch_size)
            patches.append(patch_pixels.unsqueeze(1))
        patches = torch.cat(patches, dim=1)
        return patches
```

#### 3.2.2 ConvHead

A convolutional head that processes the 2D feature map:

```python
class ConvHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, in_chans, kernel_size=3, padding=1)
        )

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.conv(features_2d)
        return out
```

#### 3.2.3 ResidualBlockHead

A head with residual connections for improved gradient flow:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class ResidualBlockHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.initial_conv = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_chans),
            ResidualBlock(in_chans),
            ResidualBlock(in_chans)
        )

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.initial_conv(features_2d)
        x = self.res_blocks(x)
        return x
```

#### 3.2.4 UNetHead

A U-Net architecture for detailed reconstruction with skip connections:

```python
class UNetHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.initial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True)
        )

        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(512+256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1),
            nn.ReLU(True)
        )

        self.final = nn.Conv2d(256, in_chans, 3, padding=1)

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        x0 = self.initial_conv(features_2d)

        x1 = self.down1(x0)
        x2 = self.down2(x1)

        btm = self.bottom(x2)

        x2_up = self.up2(btm)
        x2_cat = torch.cat([x2_up, x1], dim=1)
        x2_up = self.conv_up2(x2_cat)

        x1_up = self.up1(x2_up)
        x1_cat = torch.cat([x1_up, x0], dim=1)
        x1_up = self.conv_up1(x1_cat)

        out = self.final(x1_up)
        return out
```

### 3.3 LoRA Implementation

LoRA applies low-rank decomposition to reduce the number of trainable parameters:

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular forward pass through the original linear layer
        original_output = F.linear(x, self.weight, self.bias)
        
        # Early return if no LoRA is enabled
        if not any(self.enable_lora):
            return original_output
        
        # Initialize output with the same shape as original
        B, L, _ = x.shape
        lora_output = torch.zeros_like(original_output)
        
        # Apply LoRA for Q, K, V separately
        for i in range(3):
            if self.enable_lora[i]:
                # Get the slice corresponding to Q, K, or V
                start_idx = i * self.head_dim
                end_idx = (i + 1) * self.head_dim
                
                # Compute LoRA contribution
                dropped_x = self.lora_dropout(x)
                lora_delta = dropped_x @ torch.transpose(self.lora_A[i], 0, 1) @ torch.transpose(self.lora_B[i], 0, 1)
                lora_delta = lora_delta * self.scaling
                
                # Add to the corresponding slice
                lora_output[:, :, start_idx:end_idx] = lora_delta
        
        return original_output + lora_output
```

### 3.4 QLoRA Implementation

QLoRA combines 4-bit quantization with LoRA for even greater memory efficiency:

```python
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
```

## 4. Training Methodology

### 4.1 Base Training Loop

The core training loop follows a standard PyTorch pattern, with optimization, validation, and checkpointing:

```python
def train(self):
    """Main training loop"""
    print_gpu_stats('Initial')
    
    for epoch in range(self.config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
        
        # Training phase
        self.model.train()
        train_loss = 0
        train_psnr = 0
        
        for batch_idx, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                output = self.model(noisy)
                loss = self.criterion(output, clean)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            with torch.no_grad():
                train_loss += loss.item()
                train_psnr += calculate_psnr(output, clean)
        
        train_loss /= len(self.train_loader)
        train_psnr /= len(self.train_loader)
        
        # Validation phase
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                val_loss += loss.item()
                val_psnr += calculate_psnr(output, clean)
        
        val_loss /= len(self.val_loader)
        val_psnr /= len(self.val_loader)
        
        # Record metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_psnrs.append(train_psnr.item())
        self.val_psnrs.append(val_psnr.item())
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}dB")
        print(f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}dB")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss)
            print("Saved new best model")
        
        # Early stopping check
        if self.early_stopping(val_loss):
            print("Early stopping triggered!")
            break
```

### 4.2 Full Fine-tuning Setup

For full fine-tuning, all encoder and decoder parameters are trainable:

```python
# Set parameter requires_grad
for param in self.encoder.parameters():
    param.requires_grad = True
for param in self.head.parameters():
    param.requires_grad = True
```

### 4.3 LoRA Training Setup

For LoRA, only the LoRA matrices and head parameters are trainable:

```python
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

def setup_optimizer(self):
    """Setup optimizer with different learning rates for LoRA and head"""
    # Separate LoRA and head parameters
    lora_params = []
    head_params = []
    
    # Check if any parameters require gradients
    has_trainable_params = False
    
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            has_trainable_params = True
            if 'lora_' in name:
                lora_params.append(param)
            elif 'head.' in name:
                head_params.append(param)
    
    if not has_trainable_params:
        raise ValueError("No trainable parameters found in the model!")
    
    # Create optimizer with parameter groups
    self.optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': self.config['lora_lr']},
        {'params': head_params, 'lr': self.config['head_lr']}
    ])
```

### 4.4 QLoRA Training Setup

QLoRA is similar to LoRA but uses 4-bit quantization:

```python
def setup_optimizer(self):
    """Setup optimizer with different learning rates for QLoRA and head"""
    # Separate LoRA and head parameters
    lora_params = []
    head_params = []
    
    # Check if any parameters require gradients
    has_trainable_params = False
    
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            has_trainable_params = True
            if 'lora_' in name:
                lora_params.append(param)
            elif 'head.' in name:
                head_params.append(param)
    
    if not has_trainable_params:
        raise ValueError("No trainable parameters found in the model!")
    
    # Create optimizer with parameter groups using bitsandbytes optimizers
    self.optimizer = bnb.optim.AdamW8bit([
        {'params': lora_params, 'lr': self.config['lora_lr']},
        {'params': head_params, 'lr': self.config['head_lr']}
    ])
```

## 5. Dataset and Preprocessing

### 5.1 HySpecNet-11k Dataset

The project uses the HySpecNet-11k dataset, a large-scale benchmark for hyperspectral image analysis:

- 11,483 non-overlapping image patches of size 128 × 128 pixels
- 202 spectral bands (reduced from 224 by removing water absorption bands)
- Ground Sample Distance (GSD) of 30m
- Multiple scene types and land cover categories

### 5.2 Data Loading and Preprocessing

Data loading is handled by a custom PyTorch Dataset:

```python
class HySpecNetDataset(Dataset):
    def __init__(self, patch_paths, noise_std=0.27, stripe_intensity=0.5, mode='train'):
        """
        HySpecNet dataset for denoising task.
        
        Args:
            patch_paths: List of paths to patches
            noise_std: Standard deviation of Gaussian noise
            stripe_intensity: Intensity of stripe noise (0 to 1)
            mode: 'train', 'val', or 'test' - affects noise generation
        """
        self.patch_paths = list(patch_paths)
        self.noise_std = noise_std
        self.stripe_intensity = stripe_intensity
        self.mode = mode
        
        # Set fixed seed for validation and test sets
        if mode in ['val', 'test']:
            np.random.seed(42)
            
    def add_stripe_noise(self, data):
        """Add vertical stripe noise to the data."""
        # Get data dimensions
        num_bands, height, width = data.shape
        
        # Generate random stripe positions (40% of columns will have stripes)
        num_stripes = int(0.4 * width)
        stripe_positions = np.random.choice(width, num_stripes, replace=False)
        
        # Create stripe noise array
        stripe_noise = np.zeros_like(data)
        for pos in stripe_positions:
            # Generate random stripe intensity for each band
            stripe_values = np.random.uniform(-1, 1, num_bands) * self.stripe_intensity
            stripe_noise[:, :, pos] = stripe_values.reshape(-1, 1)
        
        return data + stripe_noise
    
    def __getitem__(self, idx):
        """Required for Dataset class"""
        try:
            data_path = self.patch_paths[idx]
            
            # Load the *-DATA.npy file
            data = np.load(str(data_path)).astype(np.float32)
            
            # Normalize each band independently BEFORE creating noisy version
            normalized_data = []
            for band in data:
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    norm_band = (band - min_val) / (max_val - min_val)
                else:
                    norm_band = np.zeros_like(band)
                normalized_data.append(norm_band)
            
            data = np.stack(normalized_data)
            
            # Create noisy version with both Gaussian and stripe noise
            gaussian_noise = np.random.normal(0, self.noise_std, data.shape)
            noisy_data = data + gaussian_noise
            
            # Add stripe noise
            noisy_data = self.add_stripe_noise(noisy_data)
            
            # Clip final result to [0, 1]
            noisy_data = np.clip(noisy_data, 0, 1)
            
            # Convert to torch tensors
            clean_data = torch.from_numpy(data).float()
            noisy_data = torch.from_numpy(noisy_data).float()
            
            return noisy_data, clean_data
            
        except Exception as e:
            print(f"Error loading {data_path}: {str(e)}")
            raise
```

## 6. Experimental Setup

### 6.1 Noise Characteristics

- Gaussian noise with σ=0.27
- Additive stripe noise on 40% of the columns, with intensity 0.5

### 6.2 Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with weight decay of 0.0001
- **Learning Rate**: 
  - Full fine-tuning: 1e-4
  - LoRA/QLoRA: Head 1e-4, LoRA matrices 1e-3
- **Batch Size**: 32 (with gradient accumulation when needed)
- **Epochs**: 200
- **Early Stopping**: Patience 20, Delta 0.01
- **Evaluation Metric**: Peak Signal-to-Noise Ratio (PSNR)
- **Dataset Split**: 80% training, 10% validation, 10% testing

### 6.3 LoRA Configuration

- LoRA rank (r): 4, 8, and 16 (tested separately)
- LoRA alpha: 8, 16, and 32 (corresponding to ranks)
- LoRA dropout: 0.1

### 6.4 QLoRA Configuration

- 4-bit quantization using NF4 format
- Same LoRA parameters as above (rank 8, alpha 16)
- Uses bitsandbytes library for quantization

## 7. Results and Analysis

### 7.1 Memory Efficiency

| Method | Total Parameters | Trainable Parameters | Memory Usage (GB) |
|--------|------------------|----------------------|-------------------|
| Full Fine-tuning | 31,787,338 | 31,787,338 | 9.8 |
| UNet only | 31,787,338 | 12,396,490 | 5.2 |
| LoRA+UNet | 31,787,338 | 294,912 (LoRA) + 12,396,490 (UNet) | 4.1 |
| QLoRA+UNet | 31,787,338 | 294,912 (LoRA) + 12,396,490 (UNet) | 2.9 |

### 7.2 Denoising Performance on HySpecNet-11k Test Set

| Method | PSNR (dB) | Processing Time (s/sample) |
|--------|-----------|----------------------------|
| QRNN3D | 18.94 | 0.418 |
| MemNet3D | 16.28 | 0.309 |
| DeNet | 18.43 | 0.160 |
| DeNet3D | 20.09 | 1.233 |
| OTVCA | 14.25 | 1.305 |
| WSRRR | 21.92 | 0.810 |
| FORPDN | 22.82 | 0.205 |
| FastHyDe | 19.39 | 82.014 |
| HyRes | 22.69 | 0.093 |
| UNet | 24.10 | 0.091 |
| Full Fine-tuning | 29.17 | 0.201 |
| UNet-Encoder LoRA | 26.90 | 0.152 |
| UNet-Encoder QLoRA | 26.88 | 0.102 |

### 7.3 Denoising Performance on EnMAP Test Set (500 Patches)

| Method | PSNR (dB) | Processing Time (s/sample) |
|--------|-----------|----------------------------|
| QRNN3D | 18.74 | 0.409 |
| MemNet3D | 14.64 | 0.308 |
| DeNet | 19.94 | 0.171 |
| DeNet3D | 20.95 | 0.584 |
| OTVCA | 13.59 | 4.357 |
| WSRRR | 22.20 | 1.603 |
| FORPDN | 22.57 | 1.542 |
| FastHyDe | 20.09 | 15.338 |
| HyRes | 21.55 | 0.225 |
| UNet | 22.97 | 0.098 |
| Full Fine-tuning | 26.39 | 0.214 |
| UNet-Encoder LoRA | 25.39 | 0.016 |
| UNet-Encoder QLoRA | 25.42 | 0.023 |

### 7.4 Visual Comparison

For a single band example, the performance comparison shows:

| Method | PSNR (dB) | MSE |
|--------|-----------|-----|
| Input Noisy Image | 12.01 | 0.0631 |
| UNet | 27.38 | 0.0018 |
| Full Fine-tuning | 31.98 | 0.0006 |
| LoRA | 31.16 | 0.0007 |
| QLoRA | 28.56 | 0.0013 |

### 7.5 Key Insights

1. Full fine-tuning achieves the highest PSNR but requires significantly more memory.
2. LoRA adaptation offers performance very close to full fine-tuning (only ~2dB lower) with dramatically fewer trainable parameters.
3. QLoRA maintains almost identical performance to LoRA with further memory reduction through quantization.
4. All our methods significantly outperform traditional denoising approaches and other deep learning methods.
5. The UNet head generally performs better than other head architectures.
6. Parameter-efficient methods show excellent generalization to the new EnMAP dataset.

## 8. Conclusions and Future Work

### 8.1 Main Findings

1. Parameter-efficient methods can successfully adapt pre-trained Vision Transformers for hyperspectral image denoising.
2. LoRA and QLoRA offer compelling alternatives to full fine-tuning, with minimal performance trade-offs.
3. The memory and computational efficiency of these methods make them suitable for deployment in resource-constrained environments.
4. Pre-trained frozen encoders provide strong feature representations for downstream tasks.
5. The UNet decoder head architecture consistently outperforms simpler alternatives.

### 8.2 Future Directions

1. Investigate other parameter-efficient tuning methods like S-LoRA and KD-LoRA.
2. Explore larger pre-trained models as encoders.
4. Extend the approach to other hyperspectral image analysis tasks beyond denoising.
5. Experiment with different noise types and combinations.

## 9. Running the Code

### 9.1 Hardware Requirements

For optimal performance, we recommend the following hardware configuration:

- **GPU**: NVIDIA GPU with at least 8GB of VRAM (16GB+ recommended for full fine-tuning)
- **CPU**: 8+ cores
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: At least 256GB free space for the dataset and model checkpoints

Note that memory requirements vary significantly by method:
- QLoRA is the most memory-efficient
- LoRA works well on 12GB GPUs
- Full fine-tuning generally requires 16GB+ GPUs for reasonable batch sizes

### 9.2 Installation and Environment Setup

#### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://git.tu-berlin.de/rsim/cv4rs-2024-winter/self-supervised-learning-for-hyperspectral-image-analysis.git
cd self-supervised-learning-for-hyperspectral-image-analysis
cd denoising/

# Create and activate Conda environment
conda create -n denoising python=3.10
conda activate denoising
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```


#### Dataset Setup

1. Download the HySpecNet-11k dataset from https://hyspecnet.rsim.berlin/
2. Extract the dataset and place it in a directory accessible to the project
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

```

### 9.3 Configuration Options

The `run_training.py` script accepts a wide range of configuration parameters:

#### General Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Training method (`full`, `lora`, or `qlora`) | Required |
| `--data_path` | Path to dataset directory | `/home/akram/dataset_download/hyspecnet-11k` |
| `--save_dir` | Directory to save results | Required |
| `--weights_path` | Path to pretrained weights | `/home/akram/Downloads/mae.pth` |
| `--resume_from` | Path to checkpoint to resume from | None |
| `--batch_size` | Batch size | 8 |
| `--num_epochs` | Number of training epochs | 200 |
| `--patience` | Early stopping patience | 20 |
| `--min_delta` | Early stopping minimum delta | 1e-4 |
| `--visualize_every` | Visualize results every N epochs | 1 |
| `--head_type` | Decoder head type (`fc`, `conv`, `residual`, `unet`) | `unet` |
| `--stripe_intensity` | Intensity of stripe noise | 0.5 |
| `--gradient_accumulation_steps` | Number of gradient accumulation steps | 2 |

#### Memory Management

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--empty_cache` | Empty CUDA cache before training | False |

#### Method-Specific Parameters

For full fine-tuning:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--learning_rate` | Learning rate | 1e-4 |

For LoRA and QLoRA:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--head_lr` | Learning rate for head | 1e-4 |
| `--lora_lr` | Learning rate for LoRA parameters | 1e-3 |
| `--lora_r` | Rank for LoRA adaptation | 8 |
| `--lora_alpha` | Alpha scaling factor for LoRA | 16 |
| `--lora_dropout` | Dropout probability for LoRA layers | 0.1 |

For QLoRA only:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--quant_type` | Quantization type for QLoRA (`fp4` or `nf4`) | `nf4` |

### 9.4 Training Examples

Use the universal training script to run different methods:

#### Basic Usage

```bash
# Full fine-tuning with UNet head
python run_training.py --method full --save_dir ./results/full_unet --head_type unet

# LoRA with UNet head and rank 8
python run_training.py --method lora --save_dir ./results/lora_r8 --head_type unet --lora_r 8 --lora_alpha 16

# QLoRA with UNet head using NF4 quantization
python run_training.py --method qlora --save_dir ./results/qlora_nf4 --head_type unet --quant_type nf4
```

#### Experiments with Different LoRA Ranks

```bash
# LoRA with rank 4
python run_training.py --method lora --save_dir ./results/lora_r4 --lora_r 4 --lora_alpha 8

# LoRA with rank 8
python run_training.py --method lora --save_dir ./results/lora_r8 --lora_r 8 --lora_alpha 16

# LoRA with rank 16
python run_training.py --method lora --save_dir ./results/lora_r16 --lora_r 16 --lora_alpha 32
```

#### Experiments with Different Decoder Heads

```bash
# Full fine-tuning with ConvHead
python run_training.py --method full --save_dir ./results/full_conv --head_type conv

# Full fine-tuning with ResidualHead
python run_training.py --method full --save_dir ./results/full_residual --head_type residual

# Full fine-tuning with FCHead
python run_training.py --method full --save_dir ./results/full_fc --head_type fc
```

#### Memory-Efficient Training

For large models or limited GPU memory:

```bash
# Reduce batch size and use gradient accumulation
python run_training.py --method lora --save_dir ./results/lora_r16 \
    --batch_size 4 --gradient_accumulation_steps 4 --empty_cache

# Use QLoRA for extreme memory efficiency
python run_training.py --method qlora --save_dir ./results/qlora_nf4 \
    --quant_type nf4 --batch_size 4 --gradient_accumulation_steps 4
```

### 9.5 Continuing Training from a Checkpoint

If your training was interrupted, you can resume from a saved checkpoint:

```bash
python run_training.py --method lora --save_dir ./results/lora_r8_continued \
    --resume_from ./results/lora_r8/checkpoint_epoch_42.pt
```

### 9.6 Monitoring and Visualization

#### During Training

The training script automatically:
- Prints training and validation loss/PSNR metrics for each epoch
- Visualizes example denoising results periodically
- Plots training and validation curves
- Saves model checkpoints
- Monitors GPU memory usage

Training metrics are saved in the results directory and include:
- Loss curves (`loss_curves.png`)
- Sample visualizations at intervals specified by `--visualize_every`
- Model checkpoints (saved when validation loss improves)

#### After Training

To visualize model results on a test set:

```bash
# Visualize full fine-tuning model results
python tests/test_set_unet.py

# Visualize LoRA model results
python tests/test_lora.py

# Visualize QLoRA model results
python tests/test_qlora.py
```

### 9.7 Testing and Evaluation

#### Testing on HySpecNet-11k Test Set

You can test your trained models on the HySpecNet-11k test set:

```bash
# Test full fine-tuning model
python tests/test_set_unet.py --checkpoint_path ./results/full_unet/checkpoint_epoch_100.pt

# Test LoRA model
python tests/test_lora.py --checkpoint_path ./results/lora_r8/checkpoint_epoch_100.pt

# Test QLoRA model
python tests/test_qlora.py --checkpoint_path ./results/qlora_nf4/checkpoint_epoch_100.pt
```

Each test script saves results to a subdirectory within the save directory:
- Metrics (PSNR, MSE, SSIM) for the entire test set
- Visualizations of sample results
- Processing time measurements

#### Testing on New Datasets (EnMAP)

You can test your trained models on the EnMAP dataset to evaluate generalization:

```bash
# Test on new dataset using LoRA model
python tests/test_lora_on_new_set.py --checkpoint_path ./results/lora_r8/checkpoint_epoch_100.pt

# Test on new dataset using QLoRA model
python tests/test_qlora_on_new_set.py --checkpoint_path ./results/qlora_nf4/checkpoint_epoch_100.pt

# Test on new dataset using full fine-tuning model
python tests/test_unet_ff_on_new_set.py --checkpoint_path ./results/full_unet/checkpoint_epoch_100.pt
```

### 9.8 Customizing Models

#### Creating Custom Head Architectures

You can create custom head architectures by extending the base classes:

```python
# In src/models/custom_head.py
import torch.nn as nn
from src.models.model import BaseHead

class MyCustomHead(BaseHead):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        
        # Define your custom architecture
        self.custom_network = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, in_chans, kernel_size=3, padding=1)
        )

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        return self.custom_network(features_2d)

# In src/models/model.py - Add to DownstreamModel.__init__
elif head_type == "custom":
    self.head = MyCustomHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
```

#### Modifying Training Procedures

You can customize the training procedure by creating a new trainer class:

```python
# In src/training/custom_trainer.py
from src.training.train_lora import LoRATrainer

class CustomLoRATrainer(LoRATrainer):
    def __init__(self, config):
        super().__init__(config)
        # Add custom initialization
        
    def train_epoch(self):
        """Custom training epoch logic"""
        # Implement custom training logic
        
    def validate(self):
        """Custom validation logic"""
        # Implement custom validation logic
```

### 9.9 Troubleshooting

#### Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use LoRA or QLoRA instead of full fine-tuning
   - Try lowering precision (e.g., use float16)
   - Make sure to use `--empty_cache` flag

2. **Slow Training**
   - Ensure you're using GPU acceleration
   - Check for CPU bottlenecks in data loading (increase `num_workers`)
   - Use mixed precision training
   - Experiment with different batch sizes

3. **Poor Performance**
   - Check learning rates (try different values)
   - Ensure model is actually training (loss should decrease)
   - Try different head architectures
   - Check dataset normalization and noise parameters
   - Increase training epochs

4. **NaN or Inf Loss**
   - Check for incorrect normalization
   - Lower learning rate
   - Check for errors in custom implementations
   - Add gradient clipping

5. **QLoRA Installation Issues**
   - Ensure bitsandbytes is properly installed for your CUDA version
   - On Windows, special setup may be required for bitsandbytes

### 9.10 Advanced Usage

#### Model Export and Deployment

You can export your trained models for deployment:

```python
# Export a trained model
import torch
from src.models import LoRADownstreamModel, load_vit_weights

# Load model
model = LoRADownstreamModel(
    img_size=128, 
    patch_size=4,
    in_chans=202,
    head_type="unet",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0
)
checkpoint = torch.load("./results/lora_r8/checkpoint_epoch_100.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Merge LoRA weights for efficiency
model.merge_lora_weights()

# Export model (TorchScript)
example_input = torch.randn(1, 202, 128, 128)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "exported_model.pt")
```

#### Batch Processing

For processing large datasets:

```python
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.models import LoRADownstreamModel, load_vit_weights

# Load model
model = LoRADownstreamModel(img_size=128, patch_size=4, in_chans=202, 
                           head_type="unet", lora_r=8, lora_alpha=16)
checkpoint = torch.load("./results/model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().cuda()

# Process files
input_dir = Path("./input_data")
output_dir = Path("./output_data")
output_dir.mkdir(exist_ok=True)

for file_path in tqdm(list(input_dir.glob("*.npy"))):
    # Load and preprocess data
    data = np.load(str(file_path)).astype(np.float32)
    
    # Normalize
    normalized_data = []
    for band in data:
        min_val, max_val = np.min(band), np.max(band)
        norm_band = (band - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(band)
        normalized_data.append(norm_band)
    data = np.stack(normalized_data)
    
    # Add noise if needed
    noisy_data = data + np.random.normal(0, 0.27, data.shape)
    
    # Convert to tensor and process
    tensor_data = torch.from_numpy(noisy_data).unsqueeze(0).cuda()
    with torch.no_grad():
        denoised = model(tensor_data).squeeze(0).cpu().numpy()
    
    # Save result
    output_path = output_dir / file_path.name.replace(".npy", "_denoised.npy")
    np.save(str(output_path), denoised)
```

## Acknowledgments

This project builds upon several key libraries and resources:

- HySpecNet-11k dataset for hyperspectral image analysis
- PyTorch and torchvision for deep learning
- Bitsandbytes library for quantization
- HyDe package for baseline comparison methods
- Vision Transformer (ViT) architecture and MAE pre-training strategy
