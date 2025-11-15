#!/usr/bin/env python3
"""
Odia-English Translation Training Script
This script handles model training only.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Import all components
from config import BertConfig, DataConfig, DEFAULT_BERT_CONFIG, DEFAULT_DATA_CONFIG
from translation_model import BertForTranslation
from tokenizer import create_tokenizer
from dataset import create_data_loaders, create_data_loaders_with_huggingface

class TranslationTrainer:
    """Trainer class for Odia-English translation model."""
    
    def __init__(self, model: BertForTranslation, tokenizer, config: BertConfig, data_config: DataConfig):
        """Initialize the trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create checkpoint directory
        os.makedirs(config.model_save_path, exist_ok=True)
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=self.config.warmup_steps
            )
            
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps - self.config.warmup_steps
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.config.warmup_steps]
            )
        else:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits, loss = self.model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                self.learning_rates.append(current_lr)
            
            # Save checkpoint periodically
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                logits, loss = self.model(
                    input_ids=batch['input_ids'],
                    token_type_ids=batch['token_type_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += loss.item()
                non_pad_tokens = (batch['labels'] != -100).sum().item()
                total_tokens += non_pad_tokens
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.val_losses.append(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Validation batches per epoch: {len(val_loader)}")
        
        total_steps = len(train_loader) * self.config.num_epochs
        self._create_scheduler(total_steps)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"  Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best_model")
                print(f"  New best model saved! (Val Loss: {self.best_val_loss:.4f})")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint("final_model")
        self.save_training_history()
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'warmup_steps': self.config.warmup_steps,
                'weight_decay': self.config.weight_decay
            },
            'timestamp': datetime.now().isoformat()
        }
        
        history_path = os.path.join(self.config.model_save_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved: {history_path}")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Odia-English Translation Training")
    parser.add_argument('--use-huggingface', action='store_true', help='Use mrSoul7766 dataset')
    parser.add_argument('--dataset', type=str, default='mrSoul7766/eng_to_odia_translation_20k', help='HF dataset name')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--config', type=str, help='Custom config file')
    
    args = parser.parse_args()
    
    # Load configurations
    config = DEFAULT_BERT_CONFIG
    data_config = DEFAULT_DATA_CONFIG
    
    print("=== Odia-English Translation Training ===")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(config)
    
    # Update vocab size
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Create model
    print("Creating model...")
    model = BertForTranslation(config)
    
    # Create trainer
    trainer = TranslationTrainer(model, tokenizer, config, data_config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    print("Creating data loaders...")
    if args.use_huggingface:
        train_loader, val_loader, test_loader = create_data_loaders_with_huggingface(
            tokenizer, data_config, config, use_huggingface=True, dataset_name=args.dataset
        )
    else:
        train_loader, val_loader, test_loader = create_data_loaders(
            tokenizer, data_config, config
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()