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

class SpectralEarthDataset(Dataset):
    def __init__(self, base_path, noise_std=0.27, stripe_intensity=0.5, mode='test', limit_samples=500):
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
                    if len(self.file_paths) >= limit_samples:  # Limit the number of samples
                        break
            if len(self.file_paths) >= limit_samples:  # Break outer loop too
                break
        
        print(f"Found {len(self.file_paths)} files in EnMAP dataset")
    
    def add_stripe_noise(self, data):
        num_bands, height, width = data.shape
        num_stripes = int(0.4 * width)
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
            with rasterio.open(file_path) as src:
                data = src.read().astype(np.float32)
                
                nodata_mask = data == -32768.0
                data[nodata_mask] = 0
                
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
                
                noisy_data = clean_data.copy()
                gaussian_noise = np.random.normal(0, self.noise_std, clean_data.shape)
                noisy_data = noisy_data + gaussian_noise
                noisy_data = self.add_stripe_noise(noisy_data)
                noisy_data = np.clip(noisy_data, 0, 1)
                
                return torch.from_numpy(noisy_data).float(), torch.from_numpy(clean_data).float()
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

def evaluate_hyde_methods(test_loader, device='cuda'):
    print("Starting HyDe methods evaluation...")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    sys.stdout.flush()
    
    results = {}
    metrics = {}

    # Initialize conventional methods
    conventional_methods = {
        'OTVCA': OTVCA(),
        'WSRRR': WSRRR(),
        'FORPDN': FORPDN_SURE(),
        'FastHyDe': FastHyDe(),
        'HyRes': HyRes(),
    }
    
    # Initialize neural network methods
    nn_methods = {
        'QRNN3D': {'arch': 'qrnn3d', 'path': '/home/akram/HyDe/pretrained-models/qrnn3d/hyde-bs16-blindSNR-gaussian-qrnn3d-l2.pth'},
        'QRNN2D': {'arch': 'qrnn2d', 'path': '/home/akram/HyDe/pretrained-models/qrnn2d/hyde-bs16-blindSNR-gaussian-qrnn2d-l2.pth'},
        'MemNet': {'arch': 'memnet', 'path': '/home/akram/HyDe/pretrained-models/memnet/hyde-bs16-blindSNR-gaussian-memnet-l2.pth'},
        'MemNet3D': {'arch': 'memnet3d', 'path': '/home/akram/HyDe/pretrained-models/memnet3d/hyde-bs16-blindSNR-gaussian-memnet3d-l2.pth'},
        'MemNet3D_64': {'arch': 'memnet3d_64', 'path': '/home/akram/HyDe/pretrained-models/memnet3d_64/memnet3d_64-gauss-l2.pth'},
        'DeNet': {'arch': 'denet', 'path': '/home/akram/HyDe/pretrained-models/denet/hyde-bs16-blindSNR-gaussian-denet-l2.pth'},
        'DeNet3D': {'arch': 'denet3d', 'path': '/home/akram/HyDe/pretrained-models/denet3d/hyde-bs16-blindSNR-gaussian-denet3d-l2.pth'},
        'MemNet_HyRes': {'arch': 'memnet_hyres', 'path': '/home/akram/HyDe/pretrained-models/memnet_hyres/hyde-bs16-blindSNR-gaussian-memnet_hyres-l2.pth'},
    }

    # Combine methods
    methods = {}
    methods.update(conventional_methods)
    
    # Add neural methods
    for name, config in nn_methods.items():
        if Path(config['path']).exists():
            try:
                methods[name] = NNInference(arch=config['arch'], pretrained_file=config['path'])
                print(f"Loaded neural method: {name}")
            except Exception as e:
                print(f"Failed to load {name}: {str(e)}")
    
    print(f"Initialized {len(methods)} methods")
    sys.stdout.flush()

    for method_name, method in methods.items():
        print(f"\nEvaluating {method_name}...")
        sys.stdout.flush()
        
        try:
            if torch.cuda.is_available():
                method = method.to(device)
                print(f"Moved {method_name} to {device}")
            
            total_psnr = 0
            total_time = 0
            n_samples = 0
            
            for batch_idx, (noisy_batch, clean_batch) in enumerate(test_loader):
                if torch.cuda.is_available():
                    noisy_batch = noisy_batch.to(device)
                    clean_batch = clean_batch.to(device)
                
                with torch.no_grad():
                    start_time = time.time()
                    
                    batch_results = []
                    for i in range(len(noisy_batch)):
                        noisy = noisy_batch[i]
                        clean = clean_batch[i]
                        
                        if method_name in nn_methods:
                            denoised = method(noisy.unsqueeze(0), band_dim=1, permute=True)
                            denoised = denoised.squeeze(0)
                        else:
                            noisy = noisy.permute(1, 2, 0)
                            if method_name == 'OTVCA':
                                denoised, _ = method(noisy, noisy.shape[-1])
                            elif method_name == 'WSRRR':
                                denoised, _ = method(noisy, rank=5)
                            else:
                                denoised = method(noisy)
                            
                            if denoised.ndim == 3:
                                denoised = denoised.permute(2, 0, 1)
                        
                        batch_results.append(denoised)
                    
                    denoised_batch = torch.stack(batch_results)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_time = time.time() - start_time
                    
                    mse = torch.mean((denoised_batch - clean_batch) ** 2)
                    psnr = 10 * torch.log10(1.0 / mse)
                    
                    total_psnr += psnr.item() * len(noisy_batch)
                    total_time += batch_time
                    n_samples += len(noisy_batch)

                print(f"\rProcessing batch {batch_idx+1}/{len(test_loader)}", end="")
                sys.stdout.flush()
            
            if n_samples > 0:
                avg_psnr = total_psnr / n_samples
                avg_time = total_time / n_samples
                
                metrics[method_name] = {
                    'PSNR': avg_psnr,
                    'Time_per_sample': avg_time
                }
                
                print(f"\n{method_name} Results:")
                print(f"Average PSNR: {avg_psnr:.2f} dB")
                print(f"Average time: {avg_time:.3f} s/sample")
                sys.stdout.flush()
            
        except Exception as e:
            print(f"Error in method {method_name}: {str(e)}")
            metrics[method_name] = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return metrics

if __name__ == "__main__":
    base_path = '/home/akram/Desktop/deeplearning1/spectral_earth_subset'
    batch_size = 8
    
    if torch.cuda.is_available():
        print("\nGPU detected!")
        print(f"Device: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    
    # Create dataset and dataloader with 500 samples limit
    dataset = SpectralEarthDataset(base_path, limit_samples=500)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    try:
        results = evaluate_hyde_methods(dataloader)
        
        print("\nFinal Results Summary:")
        print("-" * 50)
        for method_name, method_metrics in results.items():
            if method_metrics is not None:
                print(f"\n{method_name}:")
                print(f"PSNR: {method_metrics['PSNR']:.2f} dB")
                print(f"Processing time: {method_metrics['Time_per_sample']:.3f} s/sample")
                
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")