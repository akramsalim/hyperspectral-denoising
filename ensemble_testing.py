import torch
import numpy as np
from tqdm import tqdm
from utils import calculate_metrics
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_multichannel_metrics(pred, target):
    """Calculate metrics for multi-channel images by averaging across channels."""
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got pred: {pred.ndim}D, target: {target.ndim}D")
    
    # Calculate metrics for each channel and average
    num_channels = pred.shape[0]
    psnrs = []
    mses = []
    ssims = []
    
    for c in range(num_channels):
        psnr, mse, ssim = calculate_metrics(pred[c], target[c])
        psnrs.append(psnr)
        mses.append(mse)
        ssims.append(ssim)
    
    return np.mean(psnrs), np.mean(mses), np.mean(ssims)

def visualize_ensemble_results(clean, noisy, denoised, metrics, save_path):
    """Create visualization similar to training style."""
    plt.figure(figsize=(15, 5))
    
    # Global title showing improvements
    plt.suptitle(
        f'Improvements → PSNR: +{metrics["ensemble"]["psnr"]-metrics["noisy"]["psnr"]:.2f}dB | '
        f'MSE: -{metrics["noisy"]["mse"]-metrics["ensemble"]["mse"]:.6f} | '
        f'SSIM: +{metrics["ensemble"]["ssim"]-metrics["noisy"]["ssim"]:.4f}',
        y=0.95
    )
    
    # Plot clean image
    plt.subplot(131)
    plt.imshow(clean, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Clean Image')
    plt.axis('on')
    
    # Plot noisy image with metrics
    plt.subplot(132)
    plt.imshow(noisy, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f'Noisy Image\nPSNR: {metrics["noisy"]["psnr"]:.2f} dB\n'
              f'MSE: {metrics["noisy"]["mse"]:.6f}\n'
              f'SSIM: {metrics["noisy"]["ssim"]:.4f}')
    plt.axis('on')
    
    # Plot denoised image with metrics
    plt.subplot(133)
    plt.imshow(denoised, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f'Denoised Image\nPSNR: {metrics["ensemble"]["psnr"]:.2f} dB\n'
              f'MSE: {metrics["ensemble"]["mse"]:.6f}\n'
              f'SSIM: {metrics["ensemble"]["ssim"]:.4f}')
    plt.axis('on')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def ensemble_test(model, test_loader, num_ensembles=10, noise_std=0.27, device='cuda', save_dir=None):
    """
    Test model with ensemble averaging of multiple noise realizations.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    single_metrics = []
    ensemble_metrics = []
    
    with torch.no_grad():
        for batch_idx, (_, clean) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move clean images to device
            clean = clean.to(device)
            batch_size = clean.shape[0]
            
            # Storage for ensemble predictions
            ensemble_preds = torch.zeros_like(clean)
            
            # Apply different noise realizations and average predictions
            for n in range(num_ensembles):
                # Generate noisy images
                noise = torch.randn_like(clean) * noise_std
                noisy = clean + noise
                noisy = torch.clamp(noisy, 0, 1)
                
                # Get model prediction
                pred = model(noisy)
                ensemble_preds += pred
                
                # If it's the first noise realization, save metrics
                if n == 0:
                    first_noisy = noisy.clone()
                    first_pred = pred.clone()
                    
                    # Calculate metrics for single prediction
                    for i in range(batch_size):
                        psnr, mse, ssim = calculate_multichannel_metrics(
                            pred[i].cpu().numpy(),
                            clean[i].cpu().numpy()
                        )
                        single_metrics.append({
                            'psnr': psnr,
                            'mse': mse,
                            'ssim': ssim
                        })
            
            # Average the predictions
            ensemble_preds /= num_ensembles
            
            # Calculate metrics for ensemble predictions
            for i in range(batch_size):
                psnr, mse, ssim = calculate_multichannel_metrics(
                    ensemble_preds[i].cpu().numpy(),
                    clean[i].cpu().numpy()
                )
                ensemble_metrics.append({
                    'psnr': psnr,
                    'mse': mse,
                    'ssim': ssim
                })
            
            # Save visualization for first image of first batch
            if batch_idx == 0 and save_dir:
                # Calculate metrics for noisy image
                noisy_psnr, noisy_mse, noisy_ssim = calculate_multichannel_metrics(
                    first_noisy[0].cpu().numpy(),
                    clean[0].cpu().numpy()
                )
                
                # Get middle spectral band for visualization
                band_idx = clean.shape[1] // 2
                
                metrics = {
                    "noisy": {
                        "psnr": noisy_psnr,
                        "mse": noisy_mse,
                        "ssim": noisy_ssim
                    },
                    "ensemble": {
                        "psnr": ensemble_metrics[0]["psnr"],
                        "mse": ensemble_metrics[0]["mse"],
                        "ssim": ensemble_metrics[0]["ssim"]
                    }
                }
                
                visualize_ensemble_results(
                    clean=clean[0, band_idx].cpu().numpy(),
                    noisy=first_noisy[0, band_idx].cpu().numpy(),
                    denoised=ensemble_preds[0, band_idx].cpu().numpy(),
                    metrics=metrics,
                    save_path=save_dir / 'ensemble_comparison.png'
                )
    
    # Calculate average metrics
    single_avg = {
        'psnr': np.mean([m['psnr'] for m in single_metrics]),
        'mse': np.mean([m['mse'] for m in single_metrics]),
        'ssim': np.mean([m['ssim'] for m in single_metrics])
    }
    
    ensemble_avg = {
        'psnr': np.mean([m['psnr'] for m in ensemble_metrics]),
        'mse': np.mean([m['mse'] for m in ensemble_metrics]),
        'ssim': np.mean([m['ssim'] for m in ensemble_metrics])
    }
    
    # Calculate standard deviations
    single_std = {
        'psnr': np.std([m['psnr'] for m in single_metrics]),
        'mse': np.std([m['mse'] for m in single_metrics]),
        'ssim': np.std([m['ssim'] for m in single_metrics])
    }
    
    ensemble_std = {
        'psnr': np.std([m['psnr'] for m in ensemble_metrics]),
        'mse': np.std([m['mse'] for m in ensemble_metrics]),
        'ssim': np.std([m['ssim'] for m in ensemble_metrics])
    }
    
    return {
        'single': {
            'avg': single_avg,
            'std': single_std
        },
        'ensemble': {
            'avg': ensemble_avg,
            'std': ensemble_std
        },
        'improvement': {
            'psnr': ensemble_avg['psnr'] - single_avg['psnr'],
            'mse': single_avg['mse'] - ensemble_avg['mse'],
            'ssim': ensemble_avg['ssim'] - single_avg['ssim']
        }
    }