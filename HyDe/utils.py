import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False
    
    def __call__(self, val_loss):
        if not isinstance(val_loss, (float, int)):
            raise ValueError("val_loss must be a number")
            
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def print_gpu_stats(phase=''):
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        print(f"\nGPU Stats ({phase}):")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between predicted and target images."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:  # handle perfect prediction
        return 100.0
    
    # Add small epsilon to avoid numerical instability
    mse = mse + 1e-8
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    # Clip PSNR to reasonable range
    psnr = torch.clamp(psnr, min=-100, max=100)
    return psnr

def calculate_metrics(pred, target):
    """
    Calculate PSNR, MSE, and SSIM between predicted and target images.
    
    Args:
        pred (np.ndarray or torch.Tensor): Predicted image
        target (np.ndarray or torch.Tensor): Target image
        
    Returns:
        tuple: (PSNR, MSE, SSIM) metrics
    """
    # Input validation
    if pred is None or target is None:
        raise ValueError("Inputs cannot be None")
        
    # Ensure inputs are numpy arrays
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Ensure arrays have the same shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Ensure arrays are 2D (single band)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pred: {pred.ndim}D, target: {target.ndim}D")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(pred)) or np.any(np.isnan(target)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(pred)) or np.any(np.isinf(target)):
        raise ValueError("Input contains Inf values")
    
    # Ensure arrays are in valid range [0, 1]
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    
    # Calculate MSE
    mse = np.mean((pred - target) ** 2)
    
    # Calculate PSNR
    if mse < 1e-10:  # Avoid division by very small numbers
        psnr = 100.0
    else:
        max_val = 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        psnr = np.clip(psnr, -100, 100)
    
    # Calculate SSIM with error handling
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_value = ssim(target, pred,
                            data_range=1.0,
                            gaussian_weights=True,
                            use_sample_covariance=False,
                            win_size=7)
    except Exception as e:
        print(f"Warning: SSIM calculation failed - {str(e)}")
        ssim_value = 0.0
    
    return psnr, mse, ssim_value

def visualize_results(model, dataset, sample_idx=0, save_path=None):
    """Visualize denoising results with clean layout."""
    if not hasattr(model, 'eval'):
        raise ValueError("Model must be a PyTorch model")
        
    try:
        model.eval()
        noisy, clean = dataset[sample_idx]
        
        with torch.no_grad():
            device = next(model.parameters()).device
            noisy = noisy.unsqueeze(0).to(device)
            output = model(noisy)
        
        # Move tensors to CPU and convert to numpy
        clean = clean.cpu().numpy()
        noisy = noisy.squeeze(0).cpu().numpy()
        output = output.squeeze(0).cpu().numpy()
        
        # Select middle spectral band for visualization
        band_idx = clean.shape[0] // 2
        
        # Get global min and max for consistent scaling
        global_min = min(clean[band_idx].min(), noisy[band_idx].min(), output[band_idx].min())
        global_max = max(clean[band_idx].max(), noisy[band_idx].max(), output[band_idx].max())
        
        # Create figure with proper spacing
        plt.figure(figsize=(18, 6))
        plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3)
        
        # Calculate metrics
        noisy_band = np.clip(noisy[band_idx], 0, 1)
        clean_band = np.clip(clean[band_idx], 0, 1)
        output_band = np.clip(output[band_idx], 0, 1)
        
        # Calculate metrics for noisy and denoised images
        noisy_psnr, noisy_mse, noisy_ssim = calculate_metrics(noisy_band, clean_band)
        denoised_psnr, denoised_mse, denoised_ssim = calculate_metrics(output_band, clean_band)
        
        # Plot clean image without metrics
        plt.subplot(131)
        im1 = plt.imshow(clean_band, cmap='viridis', vmin=global_min, vmax=global_max)
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.title('Clean Image', fontsize=10, pad=10)
        plt.axis('on')
        plt.grid(False)
        
        # Plot noisy image with metrics
        plt.subplot(132)
        im2 = plt.imshow(noisy_band, cmap='viridis', vmin=global_min, vmax=global_max)
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.title('Noisy Image\n' +
                 f'PSNR: {noisy_psnr:.2f} dB\n' +
                 f'MSE: {noisy_mse:.6f}\n' +
                 f'SSIM: {noisy_ssim:.4f}',
                 fontsize=10, pad=10)
        plt.axis('on')
        plt.grid(False)
        
        # Plot denoised image with metrics
        plt.subplot(133)
        im3 = plt.imshow(output_band, cmap='viridis', vmin=global_min, vmax=global_max)
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        plt.title('Denoised Image\n' +
                 f'PSNR: {denoised_psnr:.2f} dB\n' +
                 f'MSE: {denoised_mse:.6f}\n' +
                 f'SSIM: {denoised_ssim:.4f}',
                 fontsize=10, pad=10)
        plt.axis('on')
        plt.grid(False)
        
        # Calculate and display improvements
        psnr_improvement = denoised_psnr - noisy_psnr
        mse_improvement = noisy_mse - denoised_mse
        ssim_improvement = denoised_ssim - noisy_ssim
        
        # Super title showing only improvements
        plt.suptitle(f'Improvements → PSNR: +{psnr_improvement:.2f}dB | MSE: -{mse_improvement:.6f} | SSIM: +{ssim_improvement:.4f}',
                    fontsize=12, y=0.98)
        
        # Save with high quality
        if save_path:
            plt.savefig(save_path, 
                       bbox_inches='tight',
                       dpi=300,
                       pad_inches=0.2,
                       facecolor='white',
                       edgecolor='none')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        plt.close()
        raise Exception(f"Visualization failed: {str(e)}")

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    if not isinstance(train_losses, (list, np.ndarray)) or not isinstance(val_losses, (list, np.ndarray)):
        raise ValueError("Losses must be lists or numpy arrays")
        
    if len(train_losses) != len(val_losses):
        raise ValueError("Train and validation losses must have the same length")
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='#2ecc71', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#e74c3c', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()