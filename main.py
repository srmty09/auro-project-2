#!/usr/bin/env python3
"""
Odia-English Translation Inference Script
This script handles model inference, translation, and model information only.
"""

import os
import sys
import argparse
import torch
import json
from typing import Optional, List
from dataclasses import dataclass

# Import components
from config import BertConfig, DataConfig, DEFAULT_BERT_CONFIG, DEFAULT_DATA_CONFIG
from translation_model import BertForTranslation
from tokenizer import create_tokenizer, ManualTokenizer

# Add ImprovedConfig class for model loading compatibility
@dataclass
class ImprovedConfig:
    """Improved configuration for model loading compatibility."""
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    max_source_length: int = 128
    max_target_length: int = 128
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    batch_size: int = 6
    num_epochs: int = 20
    warmup_steps: int = 300
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    scheduler_type: str = "cosine_with_restarts"
    cosine_restarts: int = 2
    scheduler_warmup_ratio: float = 0.1
    model_save_path: str = "./improved_checkpoints"
    save_every_n_steps: int = 500
    pretrained_model_name: str = "google/mt5-small"
    use_pretrained_t5: bool = True
    vocab_size: int = 250112

class OdiaEnglishTranslator:
    """Main class for Odia-English translation inference."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the translator."""
        # Load configurations
        if config_path and os.path.exists(config_path):
            self.config, self.data_config = self._load_config(config_path)
        else:
            self.config = DEFAULT_BERT_CONFIG
            self.data_config = DEFAULT_DATA_CONFIG
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        
        print("=== Odia-English Translation System ===")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        bert_config = BertConfig(**config_dict.get('bert_config', {}))
        data_config = DataConfig(**config_dict.get('data_config', {}))
        
        return bert_config, data_config
    
    def setup(self, corpus_file: Optional[str] = None):
        """Setup tokenizer and model."""
        print("\n=== Setting up components ===")
        
        # Create tokenizer
        print("Creating tokenizer...")
        self.tokenizer = create_tokenizer(self.config, corpus_file)
        
        # Update vocab size based on tokenizer
        actual_vocab_size = self.tokenizer.get_vocab_size()
        if actual_vocab_size != self.config.vocab_size:
            print(f"Updating vocab size from {self.config.vocab_size} to {actual_vocab_size}")
            self.config.vocab_size = actual_vocab_size
        
        # Create model
        print("Creating translation model...")
        self.model = BertForTranslation(self.config)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print("Setup completed successfully!")
    
    def load_model(self, checkpoint_path: str):
        """Load a trained model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load checkpoint with proper handling for new PyTorch versions
        try:
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
        
        # Load config from checkpoint if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Setup components if not already done
        if self.tokenizer is None:
            self.tokenizer = create_tokenizer(self.config)
        
        if self.model is None:
            self.model = BertForTranslation(self.config)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def translate(self, odia_text: str, max_length: int = 128) -> str:
        """Translate Odia text to English."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please run setup() and load_model() first")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Tokenize input
            odia_tokens = self.tokenizer.tokenize(odia_text, is_odia=True)
            odia_ids = self.tokenizer.convert_tokens_to_ids(odia_tokens, is_odia=True)
            
            # Create input tensors
            input_ids = torch.tensor([[self.config.cls_token_id] + odia_ids + [self.config.sep_token_id]], 
                                   dtype=torch.long, device=device)
            token_type_ids = torch.zeros_like(input_ids)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate translation
            generated_ids = self.model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                max_length=max_length
            )
            
            # Decode translation
            translation = self.tokenizer.decode(
                generated_ids[0].cpu().tolist(),
                skip_special_tokens=True
            )
            
            return translation.strip()
    
    def translate_batch(self, odia_texts: List[str], max_length: int = 128) -> List[str]:
        """Translate a batch of Odia texts to English."""
        translations = []
        for text in odia_texts:
            translation = self.translate(text, max_length)
            translations.append(translation)
        return translations
    
    def interactive_mode(self):
        """Run interactive translation mode."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please run setup() and load_model() first")
        
        print("\n=== Interactive Translation Mode ===")
        print("Enter Odia text to translate (type 'quit' to exit):")
        
        while True:
            try:
                odia_text = input("\nOdia: ").strip()
                
                if odia_text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not odia_text:
                    continue
                
                # Translate
                translation = self.translate(odia_text)
                print(f"English: {translation}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
    
    def model_info(self):
        """Display model information."""
        if self.model is None or self.tokenizer is None:
            print("Model not loaded. Please run setup() first.")
            return
        
        print("\n=== Model Information ===")
        print(f"Model Configuration:")
        print(f"  Vocabulary Size: {self.config.vocab_size:,}")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  Number of Layers: {self.config.num_hidden_layers}")
        print(f"  Attention Heads: {self.config.num_attention_heads}")
        print(f"  Max Sequence Length: {self.config.max_position_embeddings}")
        
        print(f"\nModel Architecture:")
        print(f"  Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        print(f"\nTokenizer Information:")
        print(f"  Combined Vocabulary Size: {self.tokenizer.get_vocab_size()}")
        print(f"  Special Tokens: {list(self.tokenizer.special_tokens.keys())}")
        
        # Test tokenization
        sample_odia = "ମୁଁ ଭଲ ଅଛି"
        sample_english = "I am fine"
        
        odia_tokens = self.tokenizer.tokenize(sample_odia, is_odia=True)
        english_tokens = self.tokenizer.tokenize(sample_english, is_odia=False)
        
        print(f"\nTokenization Examples:")
        print(f"  Odia '{sample_odia}' -> {odia_tokens}")
        print(f"  English '{sample_english}' -> {english_tokens}")
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        config_dict = {
            'bert_config': self.config.__dict__,
            'data_config': self.data_config.__dict__
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Odia-English Translation Inference")
    parser.add_argument('--mode', choices=['translate', 'interactive', 'info'], 
                       default='info', help='Operation mode')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, help='Text to translate (for translate mode)')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum translation length')
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = OdiaEnglishTranslator(args.config)
    
    if args.mode == 'info':
        # Model info mode
        translator.setup()
        translator.model_info()
        
    elif args.mode == 'translate':
        # Translation mode
        if not args.checkpoint:
            print("Error: --checkpoint required for translation mode")
            sys.exit(1)
        
        translator.setup()
        translator.load_model(args.checkpoint)
        
        if args.text:
            # Translate single text
            translation = translator.translate(args.text, args.max_length)
            print(f"Odia: {args.text}")
            print(f"English: {translation}")
        else:
            print("Error: --text required for translation mode")
            sys.exit(1)
    
    elif args.mode == 'interactive':
        # Interactive mode
        if not args.checkpoint:
            print("Error: --checkpoint required for interactive mode")
            sys.exit(1)
        
        translator.setup()
        translator.load_model(args.checkpoint)
        translator.interactive_mode()

if __name__ == "__main__":
    main()