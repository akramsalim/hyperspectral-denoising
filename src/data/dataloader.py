import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

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
    
    def __len__(self):
        return len(self.patch_paths)
    
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
            
            # Verify normalization
            assert np.min(data) >= 0 and np.max(data) <= 1, f"Data range error: [{np.min(data)}, {np.max(data)}]"
            
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
            
            # Final verification
            if torch.min(clean_data) < 0 or torch.max(clean_data) > 1:
                print(f"Warning: Clean data range: [{torch.min(clean_data):.3f}, {torch.max(clean_data):.3f}]")

            return noisy_data, clean_data
            
        except Exception as e:
            print(f"Error loading {data_path}: {str(e)}")
            raise
        
def create_dataloaders(base_path, batch_size=8, train_ratio=0.8, val_ratio=0.1, stripe_intensity=0.5):
    """Create train, validation, and test dataloaders."""
    # Get patches directory
    patches_dir = Path(base_path) / "patches"
    
    # Get all scene directories
    scene_dirs = list(patches_dir.iterdir())
    
    # Get all DATA.npy files
    all_data_files = []
    for scene_dir in scene_dirs:
        if scene_dir.is_dir():
            for patch_dir in scene_dir.iterdir():
                if patch_dir.is_dir():
                    data_file = list(patch_dir.glob("*-DATA.npy"))
                    if data_file:
                        all_data_files.append(data_file[0])
    
    # Limit dataset for quick debug
    all_data_files = all_data_files[:1000]
    print(f"Limiting dataset to {len(all_data_files)} samples for quick debug.")
    
    # Shuffle patches
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_data_files)
    
    # Calculate split indices
    n_samples = len(all_data_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split into train/val/test
    train_patches = all_data_files[:n_train]
    val_patches = all_data_files[n_train:n_train + n_val]
    test_patches = all_data_files[n_train + n_val:]
    
    print(f"Number of training samples: {len(train_patches)}")
    print(f"Number of validation samples: {len(val_patches)}")
    print(f"Number of test samples: {len(test_patches)}")
    
    # Create datasets with appropriate modes
    train_dataset = HySpecNetDataset(train_patches, mode='train', stripe_intensity=stripe_intensity)
    val_dataset = HySpecNetDataset(val_patches, mode='val', stripe_intensity=stripe_intensity)
    test_dataset = HySpecNetDataset(test_patches, mode='test', stripe_intensity=stripe_intensity)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloaders
    base_path = "/home/akram/dataset_download/hyspecnet-11k"
    train_loader, val_loader, test_loader = create_dataloaders(base_path, batch_size=16)
    
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in val loader: {len(val_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    
    # Test loading one batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"\nBatch information:")
    print(f"Noisy batch shape: {noisy_batch.shape}")
    print(f"Clean batch shape: {clean_batch.shape}")
    print(f"Noisy data range: [{noisy_batch.min():.3f}, {noisy_batch.max():.3f}]")
    print(f"Clean data range: [{clean_batch.min():.3f}, {clean_batch.max():.3f}]")