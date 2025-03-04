import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import rasterio

from src.models import QLoRADownstreamModel, load_vit_weights
from src.utils import calculate_metrics, visualize_results
from torch.utils.data import Dataset, DataLoader

class SpectralEarthDataset(Dataset):
    """Dataset class for the new EnMAP data format with noise characteristics"""
    def __init__(self, base_path, subset_type='enmap', noise_std=0.27, stripe_intensity=0.5, mode='test'):
        """
        Args:
            base_path (str): Path to spectral_earth_subset directory
            subset_type (str): Should be 'enmap' as it has 202 bands
            noise_std (float): Standard deviation of Gaussian noise
            stripe_intensity (float): Intensity of stripe noise (0 to 1)
            mode (str): Dataset mode (will be 'test' for evaluation)
        """
        # Set fixed seed for test set
        if mode in ['val', 'test']:
            np.random.seed(42)

        self.base_path = Path(base_path)
        self.subset_type = subset_type
        self.noise_std = noise_std
        self.stripe_intensity = stripe_intensity
        
        # Get all .tif files recursively
        self.file_paths = []
        subset_path = self.base_path / subset_type
        for folder in subset_path.iterdir():
            if folder.is_dir():
                for file in folder.glob('*.tif'):
                    self.file_paths.append(file)
        
        print(f"Found {len(self.file_paths)} files in {subset_type} subset")
    
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
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single image"""
        file_path = self.file_paths[idx]
        
        try:
            # Read .tif file using rasterio
            with rasterio.open(file_path) as src:
                # Read all bands
                data = src.read().astype(np.float32)  # Shape: (bands, height, width)
                
                # Verify dimensions
                if data.shape != (202, 128, 128):
                    print(f"Warning: Unexpected data shape: {data.shape}")
                    print(f"Number of bands: {src.count}")
                    print(f"Data type: {src.dtypes[0]}")
                    raise ValueError(f"Expected (202, 128, 128), got {data.shape}")
                
                # Replace nodata values with zeros
                nodata_mask = data == -32768.0
                data[nodata_mask] = 0
                
                # Normalize each band independently
                normalized_data = []
                for band in data:
                    min_val = np.min(band)
                    max_val = np.max(band)
                    if max_val > min_val:
                        norm_band = (band - min_val) / (max_val - min_val)
                    else:
                        norm_band = np.zeros_like(band)
                    normalized_data.append(norm_band)
                
                clean_data = np.stack(normalized_data)
                
                # Create noisy version exactly as in original dataloader
                noisy_data = clean_data.copy()  # Create a copy first
                
                # Add Gaussian noise
                gaussian_noise = np.random.normal(0, self.noise_std, clean_data.shape)
                noisy_data = noisy_data + gaussian_noise
                
                # Add stripe noise
                noisy_data = self.add_stripe_noise(noisy_data)
                
                # Clip final result to [0, 1]
                noisy_data = np.clip(noisy_data, 0, 1)
                
                # Convert to tensors
                clean_data = torch.from_numpy(clean_data).float()
                noisy_data = torch.from_numpy(noisy_data).float()
                
                return noisy_data, clean_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

def evaluate_on_new_dataset(model_path, config, save_dir):
    """
    Evaluate the QLoRA model on the EnMAP dataset.
    
    Args:
        model_path (str): Path to the model checkpoint
        config (dict): Configuration dictionary
        save_dir (str): Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("\nInitializing QLoRA model...")
    model = QLoRADownstreamModel(
        img_size=128,
        patch_size=4,
        in_chans=202,
        head_type=config['head_type'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        quant_type=config['quant_type'],
        compute_dtype=torch.float16
    )
    model = load_vit_weights(model, config['weights_path'])
    model = model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out the quantization-specific keys from the state dict
    filtered_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if not any(x in k for x in ['.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map', '.quant_state']):
            filtered_state_dict[k] = v
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Successfully loaded model weights (ignoring quantization state)")
    
    # Create dataset with same noise characteristics as original
    dataset = SpectralEarthDataset(
        config['data_path'], 
        subset_type='enmap',  # Only use enmap subset
        noise_std=0.27,  # Same as original
        stripe_intensity=config['stripe_intensity']  # Same as original
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    model.eval()
    total_psnr = 0
    total_mse = 0
    total_ssim = 0
    total_time = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (noisy_data, clean_data) in enumerate(tqdm(dataloader)):
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(noisy_data)
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # Save sample visualizations
            if batch_idx % 50 == 0:
                mid_band = clean_data.size(1) // 2
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Plot clean image
                im1 = axes[0].imshow(clean_data[0, mid_band].cpu().numpy(), cmap='viridis')
                axes[0].set_title('Clean Image')
                plt.colorbar(im1, ax=axes[0])
                
                # Plot noisy image
                im2 = axes[1].imshow(noisy_data[0, mid_band].cpu().numpy(), cmap='viridis')
                axes[1].set_title('Noisy Image')
                plt.colorbar(im2, ax=axes[1])
                
                # Plot denoised image
                im3 = axes[2].imshow(output[0, mid_band].cpu().numpy(), cmap='viridis')
                axes[2].set_title('Denoised Image')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{batch_idx}.png')
                plt.close()
            
            # Calculate metrics between output and clean data
            for i in range(clean_data.size(0)):
                for band_idx in range(clean_data.size(1)):
                    clean_band = clean_data[i, band_idx].cpu().numpy()
                    output_band = output[i, band_idx].cpu().numpy()
                    
                    psnr, mse, ssim = calculate_metrics(output_band, clean_band)
                    
                    total_psnr += psnr
                    total_mse += mse
                    total_ssim += ssim
                    num_samples += 1
    
    # Calculate averages
    avg_psnr = total_psnr / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    avg_time = total_time / len(dataloader)
    
    # Store results
    results = {
        'Average PSNR': f"{avg_psnr:.2f} dB",
        'Average MSE': f"{avg_mse:.6f}",
        'Average SSIM': f"{avg_ssim:.4f}",
        'Average Processing Time': f"{avg_time:.4f} seconds/batch",
        'Number of samples': num_samples,
        'Batch Size': config['batch_size']
    }

    # Save results
    with open(save_dir / 'test_results.txt', 'w') as f:
        f.write("=== QLoRA Model Test Results on EnMAP Dataset ===\n\n")
        f.write(f"Model checkpoint: {model_path}\n")
        f.write(f"QLoRA rank: {config['lora_r']}\n")
        f.write(f"QLoRA alpha: {config['lora_alpha']}\n")
        f.write(f"Quantization type: {config['quant_type']}\n")
        f.write(f"Head type: {config['head_type']}\n")
        f.write(f"Noise std: 0.27\n")
        f.write(f"Stripe intensity: {config['stripe_intensity']}\n\n")
        f.write("\n=== Results ===\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
    
    return results

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': '/home/akram/Desktop/deeplearning1/spectral_earth_subset',
        'save_dir': './qlora_new_dataset_results',
        'weights_path': '/home/akram/Downloads/mae.pth',
        'batch_size': 8,
        'head_type': 'unet',
        'stripe_intensity': 0.5,  # Same as original dataloader
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'quant_type': 'nf4'
    }
    
    # Model checkpoint path
    checkpoint_path = '/home/akram/Downloads/ssl_v9/ssl_v10_fc/ssl_v10_conv/ssl_v10_unet/ssl_v11_with_new_model/ssl_12_with_lora/UNETDecoder/ssl_15/ssl_16/ssl_17/ssl_18/qlora_results/checkpoint_epoch_186.pt'
    
    try:
        # Run evaluation
        results = evaluate_on_new_dataset(
            model_path=checkpoint_path,
            config=config,
            save_dir=config['save_dir']
        )
        
        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
                
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise