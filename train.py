import torch
from torch import nn
import torch.optim as optim
import time
import logging
from pathlib import Path
from tqdm import tqdm
from dataloader import create_dataloaders
from utils import EarlyStopping, print_gpu_stats, calculate_psnr, visualize_results, plot_training_curves
from model import DownstreamModel, load_pretrained_vit_weights  # Import updated names

class Trainer:
    def __init__(self, config):
        """Initialize trainer with configuration"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Create model
        # These parameters must match those in model.py to ensure compatible weight loading.
        self.model = DownstreamModel(img_size=128, patch_size=4, in_chans=224, head_type=config['head_type'])
        
        # Load pretrained ViT encoder weights
        self.model = load_pretrained_vit_weights(self.model, config['weights_path'])
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup training parameters
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate']
        )
        
        # Gradient accumulation setup
        self.virtual_batch_size = 32
        self.gradient_accumulation_steps = 4
        self.actual_batch_size = self.virtual_batch_size // self.gradient_accumulation_steps
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            base_path=config['data_path'],
            batch_size=self.actual_batch_size
        )
        
        # Initialize training utilities
        self.setup_training()
        
    def setup_training(self):
        """Setup training utilities"""
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_psnr = 0
        
        # Reset gradients at start of epoch
        self.optimizer.zero_grad()
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for idx, (noisy, clean) in enumerate(pbar):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    output = self.model(noisy)
                    loss = self.criterion(output, clean) / self.gradient_accumulation_steps
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                # Calculate metrics
                with torch.no_grad():
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    psnr = calculate_psnr(output, clean)
                    total_psnr += psnr.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr.item():.2f}'
                })
        
        return total_loss / len(self.train_loader), total_psnr / len(self.train_loader)
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for noisy, clean in pbar:
                    noisy = noisy.to(self.device)
                    clean = clean.to(self.device)
                    
                    output = self.model(noisy)
                    loss = self.criterion(output, clean)
                    psnr = calculate_psnr(output, clean)
                    
                    total_loss += loss.item()
                    total_psnr += psnr.item()
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'psnr': f'{psnr.item():.2f}'
                    })
        
        return total_loss / len(self.val_loader), total_psnr / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        
    def train(self):
        """Main training loop"""
        print_gpu_stats('Initial')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            train_loss, train_psnr = self.train_epoch()
            
            # Validation phase
            val_loss, val_psnr = self.validate()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}dB")
            print(f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}dB")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f"Saved new best model")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print("Early stopping triggered!")
                break
            
            # Visualize results periodically
            if (epoch + 1) % self.config['visualize_every'] == 0:
                visualize_results(
                    self.model,
                    self.val_loader.dataset,
                    save_path=self.save_dir / f'results_epoch_{epoch+1}.png'
                )
            
            # Plot training curves
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                save_path=self.save_dir / f'loss_curves.png'
            )
            
            # Monitor GPU usage
            print_gpu_stats(f'End of epoch {epoch+1}')
            
            # Optional: clear GPU cache
            torch.cuda.empty_cache()

if __name__ == "__main__":
# In train.py
    config = {
        'data_path': '/home/akram/Downloads/hyspecnet-11k',
        'save_dir': './mae_finetuning_results',
        'weights_path': '/home/akram/Downloads/mae.pth',
        'resume_from': './mae_finetuning_results/checkpoint_epoch_142.pt',  
        'batch_size': 128,
        'learning_rate': 1e-4,
        'num_epochs': 250,
        'patience': 10,
        'min_delta': 1e-4,
        'visualize_every': 5,
        'head_type': 'unet'  #  this line - can be 'fc', 'conv', 'residual', or 'unet'
    }    


    trainer = Trainer(config)
    trainer.train()
