import torch
from pathlib import Path
from model import DownstreamModel
from dataloader import create_dataloaders
from ensemble_testing import ensemble_test

# Configuration
config = {
    'data_path': '/home/akram/Downloads/hyspecnet-11k',  # Your data path
    'checkpoint_path': './mae_finetuning_results/checkpoint_epoch_142.pt',  # Your checkpoint
    'save_dir': './ensemble_results',
    'batch_size': 4,
    'num_ensembles': 10,
    'noise_std': 0.27,
    'head_type': 'unet'  # Make sure this matches what you used in training
}

# Create save directory
save_dir = Path(config['save_dir'])
save_dir.mkdir(parents=True, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DownstreamModel(img_size=128, patch_size=4, in_chans=224, head_type=config['head_type'])
checkpoint = torch.load(config['checkpoint_path'], map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Create test dataloader
_, _, test_loader = create_dataloaders(
    base_path=config['data_path'],
    batch_size=config['batch_size']
)

# Run ensemble testing
print("\nStarting ensemble testing...")
results = ensemble_test(
    model=model,
    test_loader=test_loader,
    num_ensembles=config['num_ensembles'],
    noise_std=config['noise_std'],
    device=device,
    save_dir=save_dir
)

# Print results
print("\nSingle Prediction Results:")
print(f"PSNR: {results['single']['avg']['psnr']:.2f} ± {results['single']['std']['psnr']:.2f} dB")
print(f"MSE: {results['single']['avg']['mse']:.6f} ± {results['single']['std']['mse']:.6f}")
print(f"SSIM: {results['single']['avg']['ssim']:.4f} ± {results['single']['std']['ssim']:.4f}")

print("\nEnsemble Prediction Results:")
print(f"PSNR: {results['ensemble']['avg']['psnr']:.2f} ± {results['ensemble']['std']['psnr']:.2f} dB")
print(f"MSE: {results['ensemble']['avg']['mse']:.6f} ± {results['ensemble']['std']['mse']:.6f}")
print(f"SSIM: {results['ensemble']['avg']['ssim']:.4f} ± {results['ensemble']['std']['ssim']:.4f}")

print("\nImprovements:")
print(f"PSNR: +{results['improvement']['psnr']:.2f} dB")
print(f"MSE: -{results['improvement']['mse']:.6f}")
print(f"SSIM: +{results['improvement']['ssim']:.4f}")