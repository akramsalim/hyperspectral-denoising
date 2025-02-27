import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your existing modules
from src.models import ModifiedViT, DownstreamModel, load_vit_weights
from src.data import create_dataloaders
from src.utils import calculate_metrics, visualize_results

def evaluate_on_test_set(checkpoint_path, config):
    """
    Evaluate the model on the entire test set.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        config (dict): Configuration dictionary containing necessary parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory for test results
    save_dir = Path(config['save_dir']) / 'test_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Create model and load checkpoint
    print("\nInitializing model...")
    model = DownstreamModel(
        img_size=128,
        patch_size=4,
        in_chans=202,
        head_type=config['head_type']
    )
    model = load_vit_weights(model, config['weights_path'])
    model = model.to(device)
    
    # Load the checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Rest of the function remains the same...
    # 2) Create test dataloader
    print("\nCreating test dataloader...")
    _, _, test_loader = create_dataloaders(
        base_path=config['data_path'],
        batch_size=config['batch_size'],
        stripe_intensity=config.get('stripe_intensity', 0.5)
    )
    
    # 3) Evaluate on test set
    model.eval()
    total_psnr = 0
    total_mse = 0
    total_ssim = 0
    num_samples = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(tqdm(test_loader)):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            
            # Calculate metrics for each image in batch
            for i in range(noisy.size(0)):
                # Select middle spectral band for visualization
                band_idx = clean.size(1) // 2
                clean_band = clean[i, band_idx].cpu().numpy()
                output_band = output[i, band_idx].cpu().numpy()
                noisy_band = noisy[i, band_idx].cpu().numpy()
                
                # Calculate metrics
                psnr, mse, ssim = calculate_metrics(output_band, clean_band)
                
                total_psnr += psnr
                total_mse += mse
                total_ssim += ssim
                num_samples += 1
                
                # Save some example results (e.g., every 100th sample)
                if batch_idx % 100 == 0 and i == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Plot clean image
                    im1 = axes[0].imshow(clean_band, cmap='viridis')
                    axes[0].set_title('Clean Image')
                    plt.colorbar(im1, ax=axes[0])
                    
                    # Plot noisy image
                    im2 = axes[1].imshow(noisy_band, cmap='viridis')
                    axes[1].set_title('Noisy Image')
                    plt.colorbar(im2, ax=axes[1])
                    
                    # Plot denoised image
                    im3 = axes[2].imshow(output_band, cmap='viridis')
                    axes[2].set_title(f'Denoised Image\nPSNR: {psnr:.2f}dB')
                    plt.colorbar(im3, ax=axes[2])
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / f'example_result_{batch_idx}.png')
                    plt.close()
    
    # Calculate average metrics
    avg_psnr = total_psnr / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Print and save results
    results = {
        'Average PSNR': f"{avg_psnr:.2f} dB",
        'Average MSE': f"{avg_mse:.6f}",
        'Average SSIM': f"{avg_ssim:.4f}",
        'Number of test samples': num_samples
    }
    
    print("\nTest Set Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
    # Save results to file
    with open(save_dir / 'test_results.txt', 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
    
    return results

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': '/home/akram/dataset_download/hyspecnet-11k',
        'save_dir': '/home/akram/Downloads/ssl_v9/ssl_v10_fc/ssl_v10_conv/ssl_v10_unet/ssl_v11_with_new_model/ssl_12_with_lora/UNETDecoder/ssl_15/ssl_16/ssl_17/ssl_18/mae_finetuning_results_unet_ff',
        'weights_path': '/home/akram/Downloads/mae.pth',
        'batch_size': 8,
        'head_type': 'unet',
        'stripe_intensity': 0.5
    }
    
    # Checkpoint path
    checkpoint_path = '/home/akram/Downloads/ssl_v9/ssl_v10_fc/ssl_v10_conv/ssl_v10_unet/ssl_v11_with_new_model/ssl_12_with_lora/UNETDecoder/ssl_15/ssl_16/ssl_17/ssl_18/mae_finetuning_results_unet_ff/checkpoint_epoch_246.pt'
    
    # Run evaluation
    results = evaluate_on_test_set(checkpoint_path, config)
