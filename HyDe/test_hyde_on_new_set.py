import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import rasterio
import time
import sys

# Import HyDe methods
from hyde.spatial_domain.fast_hyde import FastHyDe
from hyde.transform_domain.hyres import HyRes
from hyde.transform_domain.otvca import OTVCA
from hyde.transform_domain.wsrrr import WSRRR
from hyde.transform_domain.forpdn import FORPDN_SURE
from hyde.nn import NNInference

class SpectralEarthDatasetForHyDE(Dataset):
    """Dataset class adapted for HyDE evaluation on new data"""
    def __init__(self, base_path, noise_std=0.27, stripe_intensity=0.5, mode='test'):
        if mode in ['val', 'test']:
            np.random.seed(42)

        self.base_path = Path(base_path)
        self.noise_std = noise_std
        self.stripe_intensity = stripe_intensity
        
        # Get all .tif files from enmap directory
        self.file_paths = []
        subset_path = self.base_path / 'enmap'
        for folder in subset_path.iterdir():
            if folder.is_dir():
                for file in folder.glob('*.tif'):
                    self.file_paths.append(file)
        
        print(f"Found {len(self.file_paths)} files in EnMAP dataset")
    
    def add_stripe_noise(self, data):
        """Add vertical stripe noise using your exact parameters"""
        num_bands, height, width = data.shape
        num_stripes = int(0.4 * width)  # 40% of columns like your implementation
        stripe_positions = np.random.choice(width, num_stripes, replace=False)
        
        stripe_noise = np.zeros_like(data)
        for pos in stripe_positions:
            stripe_values = np.random.uniform(-1, 1, num_bands) * self.stripe_intensity
            stripe_noise[:, :, pos] = stripe_values.reshape(-1, 1)
        
        return data + stripe_noise
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            # Read .tif file
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)
                
                # Replace nodata values
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
                
                # Create noisy version with exact same noise as your methods
                noisy_data = clean_data.copy()
                gaussian_noise = np.random.normal(0, self.noise_std, clean_data.shape)
                noisy_data = noisy_data + gaussian_noise
                noisy_data = self.add_stripe_noise(noisy_data)
                noisy_data = np.clip(noisy_data, 0, 1)
                
                return noisy_data, clean_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

def evaluate_hyde_methods(base_path, save_dir, batch_size=8, device='cuda'):
    """Evaluate HyDE methods on the new dataset"""
    print("\nInitializing evaluation...")
    sys.stdout.flush()
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = SpectralEarthDatasetForHyDE(
        base_path=base_path,
        noise_std=0.27,  # Same as your methods
        stripe_intensity=0.5  # Same as your methods
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize methods
    methods = {
        'FastHyDe': FastHyDe(),
        'HyRes': HyRes(),
        'OTVCA': OTVCA(),
        'WSRRR': WSRRR(),
        'FORPDN': FORPDN_SURE()
    }
    
    # Add neural methods if available
    nn_configs = {
        'QRNN3D': {'arch': 'qrnn3d', 'path': '/home/akram/HyDe/pretrained-models/qrnn3d/hyde-bs16-blindSNR-gaussian-qrnn3d-l2.pth'},
        'QRNN2D': {'arch': 'qrnn2d', 'path': '/home/akram/HyDe/pretrained-models/qrnn2d/hyde-bs16-blindSNR-gaussian-qrnn2d-l2.pth'},
        'MemNet': {'arch': 'memnet', 'path': '/home/akram/HyDe/pretrained-models/memnet/hyde-bs16-blindSNR-gaussian-memnet-l2.pth'}
    }
    
    for name, config in nn_configs.items():
        if Path(config['path']).exists():
            try:
                methods[name] = NNInference(arch=config['arch'], pretrained_file=config['path'])
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {str(e)}")
    
    results = {}
    
    for method_name, method in methods.items():
        print(f"\nEvaluating {method_name}...")
        sys.stdout.flush()
        
        try:
            if torch.cuda.is_available():
                method = method.to(device)
            
            total_psnr = 0
            total_ssim = 0
            total_time = 0
            num_samples = 0
            
            for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
                print(f"Processing batch {batch_idx+1}")
                sys.stdout.flush()
                
                try:
                    if torch.cuda.is_available():
                        noisy_batch = noisy_batch.to(device)
                        clean_batch = clean_batch.to(device)
                    
                    batch_results = []
                    start_time = time.time()
                    
                    for i in range(len(noisy_batch)):
                        # Get single image
                        noisy = noisy_batch[i]
                        clean = clean_batch[i]
                        
                        # Process format based on method type
                        if method_name in nn_configs:
                            # Neural methods expect [C,H,W]
                            denoised = method(noisy.unsqueeze(0), band_dim=1, permute=True)
                            denoised = denoised.squeeze(0)
                        else:
                            # Traditional methods expect [H,W,C]
                            noisy = noisy.permute(1, 2, 0)
                            if method_name == 'OTVCA':
                                denoised, _ = method(noisy, noisy.shape[-1])
                            elif method_name == 'WSRRR':
                                denoised, _ = method(noisy, rank=5)
                            else:
                                denoised = method(noisy)
                            
                            # Convert back to [C,H,W]
                            if denoised.ndim == 3:
                                denoised = denoised.permute(2, 0, 1)
                        
                        batch_results.append(denoised)
                    
                    # Stack results
                    denoised_batch = torch.stack(batch_results)
                    batch_time = time.time() - start_time
                    
                    # Calculate metrics
                    mse = torch.mean((denoised_batch - clean_batch) ** 2)
                    psnr = 10 * torch.log10(1.0 / mse)
                    
                    total_psnr += psnr.item() * len(noisy_batch)
                    total_time += batch_time
                    num_samples += len(noisy_batch)
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            if num_samples > 0:
                avg_psnr = total_psnr / num_samples
                avg_time = total_time / num_samples
                
                results[method_name] = {
                    'PSNR': avg_psnr,
                    'Time_per_sample': avg_time
                }
                
                # Save method results
                with open(save_dir / f'{method_name}_results.txt', 'w') as f:
                    f.write(f"=== {method_name} Results ===\n")
                    f.write(f"PSNR: {avg_psnr:.2f} dB\n")
                    f.write(f"Processing time: {avg_time:.4f} s/sample\n")
                
                print(f"\n{method_name} Results:")
                print(f"PSNR: {avg_psnr:.2f} dB")
                print(f"Processing time: {avg_time:.4f} s/sample")
                
        except Exception as e:
            print(f"Error evaluating {method_name}: {str(e)}")
            results[method_name] = None
            
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save overall results
    with open(save_dir / 'all_results.txt', 'w') as f:
        f.write("=== HyDE Methods Evaluation Results ===\n\n")
        f.write("Dataset: EnMAP\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"Noise std: 0.27\n")
        f.write(f"Stripe intensity: 0.5\n\n")
        
        for method_name, method_results in results.items():
            if method_results is not None:
                f.write(f"\n{method_name}:\n")
                f.write(f"PSNR: {method_results['PSNR']:.2f} dB\n")
                f.write(f"Processing time: {method_results['Time_per_sample']:.4f} s/sample\n")
    
    return results

if __name__ == "__main__":
    base_path = '/home/akram/Desktop/deeplearning1/spectral_earth_subset'
    save_dir = './hyde_results'
    
    try:
        results = evaluate_hyde_methods(base_path, save_dir)
        
        print("\nFinal Results Summary:")
        print("-" * 50)
        for method_name, method_results in results.items():
            if method_results is not None:
                print(f"\n{method_name}:")
                print(f"PSNR: {method_results['PSNR']:.2f} dB")
                print(f"Processing time: {method_results['Time_per_sample']:.4f} s/sample")
                
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
