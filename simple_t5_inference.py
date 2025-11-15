#!/usr/bin/env python3
"""
Simple T5-based inference for Odia-English translation
This matches the model architecture from improved_kaggle_train.py
"""

import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Add ImprovedConfig for model loading compatibility
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

class SimpleT5Translator:
    """Simple T5-based translator matching the trained model."""
    
    def __init__(self, model_path="best_translation_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
        print(f"🚀 Initializing T5 Translator on {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the T5 model and tokenizer."""
        try:
            print("📦 Loading tokenizer...")
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            
            print("🤖 Loading base model...")
            self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
            
            print(f"⚡ Loading trained weights from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Map the keys from t5_model.* to the expected format
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('t5_model.'):
                        # Remove the t5_model. prefix
                        new_key = key[9:]  # Remove 't5_model.'
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                # Load the mapped state dict
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ Loaded and mapped model weights!")
            else:
                print("⚠️ No model_state_dict found, using base model")
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📝 Using base mT5 model without trained weights")
            
            # Fallback to base model
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
            self.model.to(self.device)
            self.model.eval()
    
    def translate(self, odia_text, max_length=50):
        """Translate Odia text to English."""
        try:
            # Prepare input text
            input_text = f"translate Odia to English: {odia_text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode output
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.strip()
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return f"[Error translating: {odia_text}]"

def main():
    """Test the translator."""
    translator = SimpleT5Translator()
    
    # Test translations
    test_phrases = [
        "ମୁଁ ଭଲ ଅଛି",
        "ନମସ୍କାର",
        "ତୁମର ନାମ କଣ",
        "ଆଜି ପାଗ କେମିତି ଅଛି"
    ]
    
    print("\n🔄 Testing translations:")
    print("=" * 50)
    
    for odia_text in test_phrases:
        translation = translator.translate(odia_text)
        print(f"🇮🇳 {odia_text}")
        print(f"🇬🇧 {translation}")
        print("-" * 30)

if __name__ == "__main__":
    main()
