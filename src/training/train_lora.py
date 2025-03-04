import copy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from pathlib import Path
from src.data.dataloader import create_dataloaders
from src.utils.utils import EarlyStopping, print_gpu_stats, calculate_psnr, visualize_results, plot_training_curves
from src.models.lora_model import LoRADownstreamModel, load_vit_weights


class LoRATrainer:
    def __init__(self, config):
        """Initialize trainer with configuration"""
        # Validate config first
        self.validate_config(config)
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision training
        if torch.__version__ >= '2.0.0':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Create model with LoRA
        self.model = LoRADownstreamModel(
            img_size=128,
            patch_size=4,
            in_chans=202,
            head_type=config['head_type'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout']
        )
        
        # Load pretrained weights
        self.model = load_vit_weights(self.model, config['weights_path'])
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print parameter statistics
        stats = self.model.get_parameter_groups()
        print("\n=== Parameter Statistics ===")
        print(f"Total parameters: {stats['total']:,}")
        print(f"Frozen parameters: {stats['frozen']:,}")
        print(f"LoRA trainable parameters: {stats['lora']:,}")
        print(f"Head trainable parameters: {stats['head']:,}")
        print(f"Total trainable parameters: {stats['trainable']:,}")
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            base_path=config['data_path'],
            batch_size=config['batch_size'],
            stripe_intensity=config['stripe_intensity']
        )
        
        # Initialize training utilities
        self.setup_training()
        
    @staticmethod
    def validate_config(config):
        """Validate training configuration"""
        required_fields = [
            'data_path', 'save_dir', 'weights_path', 'batch_size',
            'head_lr', 'lora_lr', 'num_epochs', 'head_type',
            'lora_r', 'lora_alpha', 'lora_dropout'
        ]
        
        # Check for required fields
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Validate numeric parameters
        if config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if config['head_lr'] <= 0:
            raise ValueError("head_lr must be positive")
        if config['lora_lr'] <= 0:
            raise ValueError("lora_lr must be positive")
        if config['num_epochs'] <= 0:
            raise ValueError("num_epochs must be positive")
        if config['lora_r'] <= 0:
            raise ValueError("lora_r must be positive")
        if config['lora_alpha'] <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0 <= config['lora_dropout'] < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
            
        # Validate paths
        if not Path(config['weights_path']).exists():
            raise FileNotFoundError(f"weights_path not found: {config['weights_path']}")
        if not Path(config['data_path']).exists():
            raise FileNotFoundError(f"data_path not found: {config['data_path']}")
            
        # Validate head type
        valid_head_types = ['fc', 'conv', 'residual', 'unet']
        if config['head_type'] not in valid_head_types:
            raise ValueError(f"Invalid head_type. Must be one of {valid_head_types}")
            
        return True
                
    def setup_optimizer(self):
        """Setup optimizer with different learning rates for LoRA and head"""
        # Separate LoRA and head parameters
        lora_params = []
        head_params = []
        
        # Check if any parameters require gradients
        has_trainable_params = False
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                has_trainable_params = True
                if 'lora_' in name:
                    lora_params.append(param)
                elif 'head.' in name:
                    head_params.append(param)
        
        if not has_trainable_params:
            raise ValueError("No trainable parameters found in the model!")
        
        # Create optimizer with parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': self.config['lora_lr']},
            {'params': head_params, 'lr': self.config['head_lr']}
        ])
        
        self.criterion = torch.nn.MSELoss()
    
    def setup_training(self):
        """Setup training utilities"""
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []  # Add PSNR tracking
        self.val_psnrs = []    # Add PSNR tracking
        self.best_val_loss = float('inf')

    def train(self):
        """Main training loop"""
        print_gpu_stats('Initial')
        
        start_epoch = 0
        if self.config.get('resume_from'):
            start_epoch = self.load_checkpoint(self.config['resume_from'])
            print(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_psnr = 0
            
            for batch_idx, (noisy, clean) in enumerate(self.train_loader):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if torch.__version__ >= '2.0.0':
                    with torch.amp.autocast('cuda'):
                        output = self.model(noisy)
                        loss = self.criterion(output, clean)
                else:
                    with torch.cuda.amp.autocast():
                        output = self.model(noisy)
                        loss = self.criterion(output, clean)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                with torch.no_grad():
                    train_loss += loss.item()
                    train_psnr += calculate_psnr(output, clean)
            
            train_loss /= len(self.train_loader)
            train_psnr /= len(self.train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_psnr = 0
            
            with torch.no_grad():
                for noisy, clean in self.val_loader:
                    noisy = noisy.to(self.device)
                    clean = clean.to(self.device)
                    
                    output = self.model(noisy)
                    loss = self.criterion(output, clean)
                    
                    val_loss += loss.item()
                    val_psnr += calculate_psnr(output, clean)
            
            val_loss /= len(self.val_loader)
            val_psnr /= len(self.val_loader)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_psnrs.append(train_psnr.item())  # Save PSNR values
            self.val_psnrs.append(val_psnr.item())      # Save PSNR values
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}dB")
            print(f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}dB")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print("Saved new best model")
            
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
            
            # Plot training curves with PSNR
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_psnrs,
                self.val_psnrs,
                save_path=self.save_dir / 'training_curves.png',
                config=self.config
            )    

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'lora_state': self.model.get_lora_state()
        }        
        # Save full checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save LoRA weights separately
        lora_path = self.save_dir / f'lora_weights_epoch_{epoch}.pt'
        self.model.save_lora_weights(lora_path)
        
        # Save merged model if requested
        if self.config.get('save_merged_model', False):
            merged_model = copy.deepcopy(self.model)
            merged_model.merge_lora_weights()
            merged_path = self.save_dir / f'merged_model_epoch_{epoch}.pt'
            torch.save(merged_model.state_dict(), merged_path)
            
        print(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and return epoch number"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Print LoRA state information
        lora_state = checkpoint.get('lora_state', None)
        if lora_state:
            print("\nLoRA Configuration:")
            print(f"Rank: {lora_state['config']['rank']}")
            print(f"Alpha: {lora_state['config']['alpha']}")
            print(f"Dropout: {lora_state['config']['dropout']}")
            print(f"\nNumber of LoRA layers: {len(lora_state['layers'])}")
            
        return checkpoint['epoch'] + 1  # Return next epoch
if __name__ == "__main__":
    config = {
        'data_path': '/home/akram/dataset_download/hyspecnet-11k',
        'save_dir': './lora_results_rank16',
        'weights_path': '/home/akram/Downloads/mae.pth',
        'resume_from': None,
        'batch_size': 16,
        'head_lr': 1e-4,
        'lora_lr': 1e-3,
        'num_epochs': 200,
        'patience': 20,
        'min_delta': 1e-4,
        'visualize_every': 1,
        'head_type': 'unet',
        'stripe_intensity': 0.5,
        'gradient_accumulation_steps': 2,
        # LoRA specific parameters
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        # Additional options
        'save_merged_model': True,  # Save merged version after training
        #'resume_from': None,  # Path to checkpoint if resuming training
    }

    # Initialize trainer
    try:
        trainer = LoRATrainer(config)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if config['resume_from']:
            start_epoch = trainer.load_checkpoint(config['resume_from'])
            print(f"Resuming training from epoch {start_epoch}")
        
        # Print initial LoRA state
        print("\nInitial LoRA State:")
        lora_state = trainer.model.get_lora_state()
        print(f"Number of LoRA layers: {len(lora_state['layers'])}")
        for layer in lora_state['layers']:
            print(f"Layer: {layer['name']}")
            print(f"  Shape: {layer['in_features']} -> {layer['out_features']}")
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise