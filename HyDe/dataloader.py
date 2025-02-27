import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class HySpecNetDataset(Dataset):
    def __init__(self, patch_paths, noise_std=0.2, stripe_intensity=0.5, mode='train'):
        """
        HySpecNet dataset for denoising task (loading *-DATA.npy files),
        now with per-band normalization for better visualization.
        
        Args:
            patch_paths (list[Path]): List of paths to patches containing *-DATA.npy
            noise_std (float): Std dev of Gaussian noise
            stripe_intensity (float): Intensity of stripe noise (0 to 1)
            mode (str): 'train', 'val', or 'test' - affects noise generation
        """
        self.patch_paths = list(patch_paths)
        self.noise_std = noise_std
        self.stripe_intensity = stripe_intensity
        self.mode = mode
        
        # Set fixed seed for validation and test sets (for reproducible noise)
        if mode in ['val', 'test']:
            np.random.seed(42)
            
    def add_stripe_noise(self, data):
        """Add vertical stripe noise to the data array shape [C, H, W]."""
        num_bands, height, width = data.shape
        
        # 40% of columns will have stripes
        num_stripes = int(0.5 * width)
        stripe_positions = np.random.choice(width, num_stripes, replace=False)
        
        stripe_noise = np.zeros_like(data)
        for pos in stripe_positions:
            # random stripe intensity for each band
            stripe_values = np.random.uniform(-1, 1, num_bands) * self.stripe_intensity
            stripe_noise[:, :, pos] = stripe_values.reshape(-1, 1)
        
        return data + stripe_noise
    
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        if self.mode in ['val', 'test']:
            np.random.seed(42 + idx)
        data_path = self.patch_paths[idx]

        # 1) Load the *-DATA.npy file
        #    e.g. path = /.../scene_01/patch_000/patch_000-DATA.npy
        try:
            data = np.load(str(data_path)).astype(np.float32)
            # data.shape = (C=202, H, W)
            
            # 2) Per-band min-max normalization
            #    This helps improve contrast for each band individually.
            #    Implementation: for i in range(C):
            #        min_val = data[i].min()
            #        max_val = data[i].max()
            #        if max_val>min_val: data[i] = (data[i] - min_val) / (max_val - min_val)
            #        else: data[i] = 0
            norm_data = []
            for c in range(data.shape[0]):
                band = data[c]
                band_min = band.min()
                band_max = band.max()
                if band_max > band_min:
                    band = (band - band_min) / (band_max - band_min)
                else:
                    # If the band is flat (all same value), just fill with zeros
                    band = np.zeros_like(band)
                norm_data.append(band)
            
            data = np.stack(norm_data, axis=0)  # shape still (202,H,W)
            
            # 3) Quick sanity check
            #    data now *should* be in [0,1], band by band
            if np.min(data) < 0 or np.max(data) > 1:
                raise ValueError(f"After normalization, data out of [0,1], range=({data.min():.3f}, {data.max():.3f}) in {data_path}")
            
            # 4) Create noisy version: Gaussian + stripe
            gaussian_noise = np.random.normal(0, self.noise_std, data.shape)
            noisy_data = data + gaussian_noise
            noisy_data = self.add_stripe_noise(noisy_data)
            
            # 5) Clip to [0,1]
            noisy_data = np.clip(noisy_data, 0, 1)
            
            # 6) Convert to torch tensors
            clean_data = torch.from_numpy(data).float()       # shape [C,H,W]
            noisy_data = torch.from_numpy(noisy_data).float() # shape [C,H,W]

            if self.mode in ['val', 'test']:
                np.random.seed(None)
            
            return noisy_data, clean_data
        
        except Exception as e:
            print(f"Error loading {data_path}: {str(e)}")
            raise

def create_dataloaders(base_path, batch_size=8, train_ratio=0.8, val_ratio=0.1, stripe_intensity=0.5):
    """
    Creates train, validation, and test dataloaders from *-DATA.npy patches.
    
    Args:
        base_path (str or Path): Path to the directory containing subfolders of patches.
        batch_size (int): Batch size for training.
        train_ratio (float): ratio of data to go in train set
        val_ratio (float): ratio of data to go in val set
        stripe_intensity (float): Stripe noise intensity
    """
    # 1) Gather all *-DATA.npy from subdirectories
    patches_dir = Path(base_path) / "patches"
    print(f"\nSearching for patches in: {patches_dir}")
    print(f"Directory exists: {patches_dir.exists()}")

    all_data_files = []
    for scene_dir in patches_dir.iterdir():
        if scene_dir.is_dir():
            for patch_dir in scene_dir.iterdir():
                if patch_dir.is_dir():
                    data_file = list(patch_dir.glob("*-DATA.npy"))
                    if data_file:
                        all_data_files.append(data_file[0])

    print(f"Found {len(all_data_files)} total DATA.npy files")

    # 2) OPTIONAL: if you want to limit dataset for quick debug:
    all_data_files = all_data_files[:1000]
    print(f"Limiting dataset to {len(all_data_files)} samples for quick debug.")

    # 3) Shuffle
    np.random.seed(42)
    np.random.shuffle(all_data_files)

    # 4) Split into train/val/test
    n_samples = len(all_data_files)
    n_train = int(n_samples * train_ratio)
    n_val   = int(n_samples * val_ratio)

    train_patches = all_data_files[:n_train]
    val_patches   = all_data_files[n_train:n_train + n_val]
    test_patches  = all_data_files[n_train + n_val:]
    
    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_patches)}")
    print(f"Validation samples: {len(val_patches)}")
    print(f"Test samples: {len(test_patches)}")

    # (Optional) If you want to ensure a "good patch" is in val set:
    good_path_str = "ENMAP01-____L2A-DT0000005103_20221107T014820Z_003_V010110_20221117T002051Z-Y04160543_X01780305-DATA.npy"
    for i, pth in enumerate(train_patches):
        if good_path_str in str(pth):
            print(f"Found good patch in TRAIN at index={i}. Moving it to VAL.")
            val_patches.append(train_patches.pop(i))
            break
    for i, pth in enumerate(test_patches):
        if good_path_str in str(pth):
            print(f"Found good patch in TEST at index={i}. Moving it to VAL.")
            val_patches.append(test_patches.pop(i))
            break

    # 5) Create Datasets
    train_dataset = HySpecNetDataset(train_patches, mode='train', stripe_intensity=stripe_intensity)
    val_dataset   = HySpecNetDataset(val_patches,   mode='val',   stripe_intensity=stripe_intensity)
    test_dataset  = HySpecNetDataset(test_patches,  mode='test',  stripe_intensity=stripe_intensity)

    # 6) Create Dataloaders
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
    # Quick test
    base_path = "/home/akram/dataset_download/hyspecnet-11k"
    train_loader, val_loader, test_loader = create_dataloaders(base_path, batch_size=8)

    print(f"\nTrain loader has {len(train_loader)} batches.")
    print(f"Val loader has   {len(val_loader)} batches.")
    print(f"Test loader has  {len(test_loader)} batches.")

    # Test loading one batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"Noisy batch shape: {noisy_batch.shape}  [should be (B, 202, H, W)]")
    print(f"Clean batch shape: {clean_batch.shape}")
    print(f"Noisy data range: [{noisy_batch.min():.3f}, {noisy_batch.max():.3f}]")
    print(f"Clean data range: [{clean_batch.min():.3f}, {clean_batch.max():.3f}]")
