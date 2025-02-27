import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Import your existing modules
from src.models import QLoRADownstreamModel, load_vit_weights
from src.data import create_dataloaders
from src.utils import calculate_metrics, visualize_results

def evaluate_on_test_set(checkpoint_path, config):
    """
    Evaluate the QLoRA model on the entire test set.
    
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
    
    # Load the checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both regular checkpoints and merged models
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Print QLoRA state information if available
        if 'qlora_state' in checkpoint:
            qlora_state = checkpoint['qlora_state']
            print("\nQLoRA Configuration:")
            print(f"Rank: {qlora_state['config']['rank']}")
            print(f"Alpha: {qlora_state['config']['alpha']}")
            print(f"Dropout: {qlora_state['config']['dropout']}")
            print(f"Quantization Type: {qlora_state['config']['quant_type']}")
            print(f"Number of QLoRA layers: {len(qlora_state['layers'])}")
    else:
        # Assume it's a merged model state dict
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded merged model weights")
    
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
    total_time = 0
    num_samples = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(tqdm(test_loader)):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(noisy)
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # Calculate metrics for each image in batch
            for i in range(noisy.size(0)):
                # Calculate metrics across all spectral bands
                for band_idx in range(clean.size(1)):
                    clean_band = clean[i, band_idx].cpu().numpy()
                    output_band = output[i, band_idx].cpu().numpy()
                    noisy_band = noisy[i, band_idx].cpu().numpy()
                    
                    # Calculate metrics
                    psnr, mse, ssim = calculate_metrics(output_band, clean_band)
                    
                    total_psnr += psnr
                    total_mse += mse
                    total_ssim += ssim
                    num_samples += 1
                
                # Save example results (e.g., every 100th sample)
                if batch_idx % 100 == 0 and i == 0:
                    # Choose middle spectral band for visualization
                    mid_band = clean.size(1) // 2
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Plot clean image
                    im1 = axes[0].imshow(clean[i, mid_band].cpu().numpy(), cmap='viridis')
                    axes[0].set_title('Clean Image')
                    plt.colorbar(im1, ax=axes[0])
                    
                    # Plot noisy image
                    im2 = axes[1].imshow(noisy[i, mid_band].cpu().numpy(), cmap='viridis')
                    axes[1].set_title('Noisy Image')
                    plt.colorbar(im2, ax=axes[1])
                    
                    # Plot denoised image
                    im3 = axes[2].imshow(output[i, mid_band].cpu().numpy(), cmap='viridis')
                    axes[2].set_title(f'Denoised Image (QLoRA)\nPSNR: {psnr:.2f}dB')
                    plt.colorbar(im3, ax=axes[2])
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / f'example_result_{batch_idx}.png')
                    plt.close()
    
    # Calculate average metrics
    avg_psnr = total_psnr / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    avg_time = total_time / len(test_loader)  # Average time per batch
    
    # Print and save results
    results = {
        'Average PSNR': f"{avg_psnr:.2f} dB",
        'Average MSE': f"{avg_mse:.6f}",
        'Average SSIM': f"{avg_ssim:.4f}",
        'Average Processing Time': f"{avg_time:.4f} seconds/batch",
        'Number of test samples': num_samples,
        'Batch Size': config['batch_size']
    }
    
    print("\nTest Set Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
    # Save results to file
    with open(save_dir / 'test_results.txt', 'w') as f:
        f.write("=== QLoRA Model Test Results ===\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"QLoRA rank: {config['lora_r']}\n")
        f.write(f"QLoRA alpha: {config['lora_alpha']}\n")
        f.write(f"Quantization type: {config['quant_type']}\n")
        f.write(f"Head type: {config['head_type']}\n\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
    
    return results

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': '/home/akram/dataset_download/hyspecnet-11k',
        'save_dir': './qlora_test_results',
        'weights_path': '/home/akram/Downloads/mae.pth',
        'batch_size': 8,
        'head_type': 'unet',
        'stripe_intensity': 0.5,
        # QLoRA specific parameters
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'quant_type': 'nf4'
    }
    
    # Checkpoint path - can be either regular checkpoint or merged model
    checkpoint_path = '/home/akram/Downloads/ssl_v9/ssl_v10_fc/ssl_v10_conv/ssl_v10_unet/ssl_v11_with_new_model/ssl_12_with_lora/UNETDecoder/ssl_15/ssl_16/ssl_17/ssl_18/qlora_results/checkpoint_epoch_186.pt'
    # Alternative: use merged model
    # checkpoint_path = './qlora_results/merged_model_epoch_199.pt'
    
    try:
        # Run evaluation
        results = evaluate_on_test_set(checkpoint_path, config)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
