#!/usr/bin/env python3
"""
Universal training script for running different training configurations.

This script allows running any of the three training methods:
1. Full fine-tuning (Trainer)
2. LoRA fine-tuning (LoRATrainer)
3. QLoRA fine-tuning (QLoRATrainer)

with various configuration options without modifying the source code.
"""

import argparse
import os
# Set this environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from pathlib import Path

# Import trainers
from src.training import Trainer, LoRATrainer, QLoRATrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script for MAE denoising models")
    
    # General arguments
    parser.add_argument("--method", type=str, required=True, choices=["full", "lora", "qlora"],
                        help="Training method: full, lora, or qlora")
    parser.add_argument("--data_path", type=str, default="/home/akram/dataset_download/hyspecnet-11k",
                        help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--weights_path", type=str, default="/home/akram/Downloads/mae.pth",
                        help="Path to pretrained weights")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=8,  # Reduced default batch size
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Early stopping minimum delta")
    parser.add_argument("--visualize_every", type=int, default=1,
                        help="Visualize results every N epochs")
    parser.add_argument("--head_type", type=str, default="unet", 
                        choices=["fc", "conv", "residual", "unet"],
                        help="Type of head to use")
    parser.add_argument("--stripe_intensity", type=float, default=0.5,
                        help="Intensity of stripe noise")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    
    # Memory management
    parser.add_argument("--empty_cache", action="store_true",
                        help="Empty CUDA cache before training")
    
    # Full fine-tuning specific arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for full fine-tuning")
    
    # LoRA and QLoRA specific arguments
    parser.add_argument("--head_lr", type=float, default=1e-4,
                        help="Learning rate for head in LoRA/QLoRA")
    parser.add_argument("--lora_lr", type=float, default=1e-3,
                        help="Learning rate for LoRA parameters")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank for LoRA adaptation")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout probability for LoRA layers")
    
    # QLoRA specific arguments
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["fp4", "nf4"],
                        help="Quantization type for QLoRA (fp4 or nf4)")
    
    return parser.parse_args()

def main():
    """Main function to run training."""
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Empty CUDA cache if requested
    if args.empty_cache and torch.cuda.is_available():
        print("Emptying CUDA cache...")
        torch.cuda.empty_cache()
    
    # Construct configuration based on method
    if args.method == "full":
        config = {
            'data_path': args.data_path,
            'save_dir': args.save_dir,
            'weights_path': args.weights_path,
            'resume_from': args.resume_from,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'visualize_every': args.visualize_every,
            'head_type': args.head_type,
            'stripe_intensity': args.stripe_intensity,
            'gradient_accumulation_steps': args.gradient_accumulation_steps
        }
        trainer = Trainer(config)
        
    elif args.method == "lora":
        config = {
            'data_path': args.data_path,
            'save_dir': args.save_dir,
            'weights_path': args.weights_path,
            'resume_from': args.resume_from,
            'batch_size': args.batch_size,
            'head_lr': args.head_lr,
            'lora_lr': args.lora_lr,
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'visualize_every': args.visualize_every,
            'head_type': args.head_type,
            'stripe_intensity': args.stripe_intensity,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'save_merged_model': True,
            'gradient_accumulation_steps': args.gradient_accumulation_steps
        }
        trainer = LoRATrainer(config)
        
    elif args.method == "qlora":
        config = {
            'data_path': args.data_path,
            'save_dir': args.save_dir,
            'weights_path': args.weights_path,
            'resume_from': args.resume_from,
            'batch_size': args.batch_size,
            'head_lr': args.head_lr,
            'lora_lr': args.lora_lr,
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'visualize_every': args.visualize_every,
            'head_type': args.head_type,
            'stripe_intensity': args.stripe_intensity,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'quant_type': args.quant_type,
            'gradient_accumulation_steps': args.gradient_accumulation_steps
        }
        trainer = QLoRATrainer(config)
    
    # Print GPU memory stats before training
    if torch.cuda.is_available():
        print(f"\nGPU Memory Before Training:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Log the configuration
    print(f"\n{'='*20} Training Configuration {'='*20}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print('='*60)
    
    # Start training
    try:
        trainer.train()
        print(f"\nTraining completed successfully. Results saved to {args.save_dir}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()