import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.utils import calculate_psnr  # This is in your utils.py
from src.data.dataloader import create_dataloaders
from src.models.model import DownstreamModel, load_vit_weights

# Set paths
checkpoint_dir = ".//home/akram/Downloads/ssl_v9/ssl_v10_fc/ssl_v10_conv/ssl_v10_unet/ssl_v11_with_new_model/ssl_12_with_lora/UNETDecoder/ssl_15/ssl_16/ssl_17/ssl_18/mae_finetuning_results_unet_ff/checkpoint_epoch_246.pt"  # Folder where checkpoints are stored
checkpoint_epochs = [f"{checkpoint_dir}/checkpoint_epoch_{i}.pt" for i in range(1, 251)]

# Load dataset
_, val_loader, _ = create_dataloaders(base_path="/home/akram/dataset_download/hyspecnet-11k", batch_size=8)

# Initialize Model
model = DownstreamModel(img_size=128, patch_size=4, in_chans=202, head_type="unet")
model = model.cuda()
model.eval()

# Store PSNR values
psnr_values = []
epochs = []

for checkpoint_path in checkpoint_epochs:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        total_psnr = 0
        count = 0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.cuda(), clean.cuda()
                output = model(noisy)
                
                psnr = calculate_psnr(output, clean)
                total_psnr += psnr.item()
                count += 1
        
        avg_psnr = total_psnr / count
        psnr_values.append(avg_psnr)
        epochs.append(checkpoint['epoch'])

        print(f"Epoch {checkpoint['epoch']}: PSNR = {avg_psnr:.2f} dB")
    
    except Exception as e:
        print(f"Skipping checkpoint {checkpoint_path}: {str(e)}")

# Plot PSNR vs. Epochs
plt.figure(figsize=(8,5))
plt.plot(epochs, psnr_values, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs. Epochs during Fine-Tuning")
plt.grid(True)
plt.savefig("psnr_vs_epochs.png", dpi=300)
plt.show()
