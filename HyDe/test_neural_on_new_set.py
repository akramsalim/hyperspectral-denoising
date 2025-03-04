import torch
import torch.nn.functional as F
import time
import sys
import numpy as np
from pathlib import Path
import rasterio
from torch.utils.data import Dataset, DataLoader

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
                    if len(self.file_paths) >= limit_samples:
                        break
            if len(self.file_paths) >= limit_samples:
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

def evaluate_neural_methods(test_loader, device='cuda'):
    print("Starting Neural Methods evaluation...")
    sys.stdout.flush()

    metrics = {}

    # Ensure these names & chunk sizes match your training:
    chunk_size_map = {
        'QRNN2D': 10,
        'MemNet': 4,
        'MemNet_HyRes': 4,
        'DeNet': 10
    }

    nn_methods = {
        'QRNN3D': {
            'arch': 'qrnn3d',
            'path': '/home/akram/HyDe/pretrained-models/qrnn3d/hyde-bs16-blindSNR-gaussian-qrnn3d-l2.pth',
            'type': '3d'
        },
        'QRNN2D': {
            'arch': 'qrnn2d',
            'path': '/home/akram/HyDe/pretrained-models/qrnn2d/hyde-bs16-blindSNR-gaussian-qrnn2d-l2.pth',
            'type': '2d'
        },
        'MemNet': {
            'arch': 'memnet',
            'path': '/home/akram/HyDe/pretrained-models/memnet/hyde-bs16-blindSNR-gaussian-memnet-l2.pth',
            'type': '2d'
        },
        'MemNet3D': {
            'arch': 'memnet3d',
            'path': '/home/akram/HyDe/pretrained-models/memnet3d/hyde-bs16-blindSNR-gaussian-memnet3d-l2.pth',
            'type': '3d'
        },
        'DeNet': {
            'arch': 'denet',
            'path': '/home/akram/HyDe/pretrained-models/denet/hyde-bs16-blindSNR-gaussian-denet-l2.pth',
            'type': '2d'
        },
        'DeNet3D': {
            'arch': 'denet3d',
            'path': '/home/akram/HyDe/pretrained-models/denet3d/hyde-bs16-blindSNR-gaussian-denet3d-l2.pth',
            'type': '3d'
        },
        'MemNet_HyRes': {
            'arch': 'memnet_hyres',
            'path': '/home/akram/HyDe/pretrained-models/memnet_hyres/hyde-bs16-blindSNR-gaussian-memnet_hyres-l2.pth',
            'type': '2d'
        }
    }

    methods = {}
    for name, config in nn_methods.items():
        try:
            model = NNInference(arch=config['arch'], pretrained_file=config['path'])
            methods[name] = {'model': model, 'type': config['type']}
            print(f"Loaded neural method: {name} ({config['type']})")
        except Exception as e:
            print(f"Failed to load {name}: {str(e)}")

    print(f"Initialized {len(methods)} methods")
    sys.stdout.flush()

    for method_name, method_info in methods.items():
        print(f"\nEvaluating {method_name}...")
        sys.stdout.flush()

        total_psnr = 0.0
        total_time = 0.0
        n_samples = 0

        try:
            model = method_info['model']
            model_type = method_info['type']

            if torch.cuda.is_available():
                model = model.to(device)
                print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                sys.stdout.flush()

            for batch_idx, (noisy_batch, clean_batch) in enumerate(test_loader):
                print(f"\rProcessing batch {batch_idx+1}/{len(test_loader)}", end="")
                sys.stdout.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    noisy_batch = noisy_batch.to(device, dtype=torch.float32)
                    clean_batch = clean_batch.to(device, dtype=torch.float32)

                with torch.no_grad():
                    start_time = time.time()

                    if model_type == '3d':
                        batch_3d = noisy_batch.permute(0, 2, 3, 1).unsqueeze(1)
                        denoised_3d = model(batch_3d, band_dim=4, permute=False)
                        denoised_3d = denoised_3d.squeeze(1).permute(0, 3, 1, 2)
                        denoised_batch = denoised_3d
                    else:
                        group_size = chunk_size_map.get(method_name, 10)
                        denoised_batch = chunk_and_pad_2d(model, noisy_batch, group_size)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_time = time.time() - start_time

                    # MSE / PSNR
                    mse = torch.mean((denoised_batch - clean_batch) ** 2)
                    psnr = 10 * torch.log10(1.0 / mse)

                    total_psnr += psnr.item() * len(noisy_batch)
                    total_time += batch_time
                    n_samples += len(noisy_batch)

                if batch_idx == 0:
                    print(f"\nFirst batch completed for {method_name}")
                    print(f"Batch PSNR: {psnr.item():.2f} dB")
                    print(f"Batch processing time: {batch_time:.3f} seconds")
                    sys.stdout.flush()

            if n_samples > 0:
                avg_psnr = total_psnr / n_samples
                avg_time = total_time / n_samples
                metrics[method_name] = {
                    'PSNR': avg_psnr,
                    'Time_per_sample': avg_time
                }

                print(f"\n{method_name} Final Results:")
                print(f"Average PSNR: {avg_psnr:.2f} dB")
                print(f"Average processing time: {avg_time:.3f} seconds/sample")
                sys.stdout.flush()

        except Exception as e:
            print(f"Error in method {method_name}: {str(e)}")
            if torch.cuda.is_available():
                print(f"Final memory state: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                torch.cuda.empty_cache()
            sys.stdout.flush()
            metrics[method_name] = None

    return metrics

def chunk_and_pad_2d(model, noisy_batch, group_size):
    """
    Splits the batch (B, C, H, W) into chunks of size 'group_size' in the channel dimension.
    Zero-pads leftover channels if needed, then un-pads the output so final result has original #channels.
    """
    B, C, H, W = noisy_batch.shape
    denoised_bands = []
    start_idx = 0
    while start_idx < C:
        end_idx = start_idx + group_size
        band_slice = noisy_batch[:, start_idx:end_idx, :, :]
        leftover = band_slice.size(1)

        if leftover < group_size:
            # zero-pad to group_size
            pad_amt = group_size - leftover
            padded_slice = F.pad(band_slice, (0, 0, 0, 0, 0, pad_amt))

            denoised_padded = model(padded_slice, band_dim=1, permute=False)
            # un-pad
            denoised_slice = denoised_padded[:, :leftover, :, :]
        else:
            denoised_slice = model(band_slice, band_dim=1, permute=False)

        denoised_bands.append(denoised_slice)
        start_idx = end_idx

    # Concatenate along channel dim => final shape is [B, C, H, W]
    denoised_batch = torch.cat(denoised_bands, dim=1)
    return denoised_batch

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
        metrics = evaluate_neural_methods(dataloader)
        
        print("\nFinal Results Summary:")
        print("-" * 50)
        for method_name, method_metrics in metrics.items():
            if method_metrics is not None:
                print(f"\n{method_name}:")
                print(f"PSNR: {method_metrics['PSNR']:.2f} dB")
                print(f"Processing time: {method_metrics['Time_per_sample']:.3f} s/sample")
                
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")