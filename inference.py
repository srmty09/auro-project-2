

import os
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import warnings
warnings.filterwarnings('ignore')

class T5TranslationModel(nn.Module):
    
    def __init__(self, pretrained_model_name="google/mt5-small"):
        super().__init__()
        self.t5_model = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32
        )
    
    def forward(self, input_ids, labels=None):
        return self.t5_model(input_ids=input_ids, labels=labels)
    
    def generate(self, input_ids, **kwargs):
        return self.t5_model.generate(input_ids, **kwargs)

class OdiaTranslator:
    
    def __init__(self, model_path="t5_weights_only.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.weights_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        print("Loading Odia-English Translation Model")
        print(f"Device: {self.device}")
        

        try:
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            print("MT5Tokenizer loaded")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            return
        

        try:
            self.model = T5TranslationModel('google/mt5-small')
            print("Base model loaded")
        except Exception as e:
            print(f"Failed to load base model: {e}")
            return
        

        if os.path.exists(self.model_path):
            try:
                print(f"Loading weights from: {self.model_path}")
                

                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"Found {len(state_dict)} model parameters")
                    

                    missing_keys, unexpected_keys = self.model.t5_model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                    
                    self.weights_loaded = True
                    print("Trained weights loaded successfully")
                    

                    try:
                        if isinstance(checkpoint, dict):
                            if 'current_epoch' in checkpoint:
                                print(f"Epochs trained: {checkpoint['current_epoch']}")
                            if 'best_val_loss' in checkpoint:
                                print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
                    except:
                        pass
                
                else:
                    print("No model_state_dict found in checkpoint")
                    
            except Exception as e:
                print(f"Failed to load weights: {e}")
                print("Using base model weights")
        else:
            print(f"Model file not found: {self.model_path}")
            print("Using base model weights")
        

        self.model.to(self.device)
        self.model.eval()
        
        if self.weights_loaded:
            print("Model ready with trained weights")
        else:
            print("Model ready with base weights only")
    
    def translate(self, odia_text):
        if not self.model or not self.tokenizer:
            return "Model not loaded properly"
        

        input_text = f"translate Odia to English: {odia_text}"
        

        input_ids = self.tokenizer.encode(
            input_text,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=50,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0
                )
            

            translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            

            translation = translation.strip()
            

            if translation.startswith("translate Odia to English:"):
                translation = translation[26:].strip()
            
            return translation
            
        except Exception as e:
            return f"Translation error: {e}"
    
    def batch_translate(self, odia_texts):
        return [self.translate(text) for text in odia_texts]
    
    def test_model(self):
        test_examples = [
            {"odia": "ମୁଁ ଭଲ ଅଛି", "english": "I am fine"},
            {"odia": "ତୁମର ନାମ କଣ", "english": "What is your name"},
            {"odia": "ଆଜି ଆବହାୱା ଭଲ", "english": "The weather is good today"},
            {"odia": "ମୁଁ ଭାତ ଖାଉଛି", "english": "I am eating rice"},
            {"odia": "ସେ ସ୍କୁଲକୁ ଯାଉଛି", "english": "He is going to school"},
            {"odia": "ଧନ୍ୟବାଦ", "english": "Thank you"},
            {"odia": "ତୁମେ କେମିତି ଅଛ", "english": "How are you"},
            {"odia": "ଆମେ ବନ୍ଧୁ", "english": "We are friends"}
        ]
        
        print(f"\n{'='*60}")
        print("TESTING MODEL")
        print(f"{'='*60}")
        
        total_tests = len(test_examples)
        for i, example in enumerate(test_examples, 1):
            odia_text = example['odia']
            expected_english = example['english']
            
            print(f"\nTest {i}/{total_tests}:")
            print(f"   Odia:      {odia_text}")
            print(f"   Expected:  {expected_english}")
            
            generated_english = self.translate(odia_text)
            print(f"   Generated: {generated_english}")
            

            if generated_english and not generated_english.startswith("❌"):
                gen_words = set(generated_english.lower().split())
                exp_words = set(expected_english.lower().split())
                if gen_words and exp_words:
                    intersection = gen_words.intersection(exp_words)
                    similarity = len(intersection) / max(len(gen_words), len(exp_words))
                    if similarity > 0.3:
                        print(f"   Good match")
                    else:
                        print(f"   Partial match")
                else:
                    print(f"   Poor match")
            else:
                print(f"   Translation failed")
        
        print(f"\n{'='*60}")
        print("Testing completed")


_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = OdiaTranslator()
    return _translator

def quick_translate(odia_text):
    translator = get_translator()
    return translator.translate(odia_text)

if __name__ == "__main__":
    translator = OdiaTranslator()
    
    translator.test_model()
    
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("Type Odia text to translate (or 'quit' to exit)")
    print(f"{'='*60}")
    
    while True:
        try:
            odia_input = input("\nEnter Odia text: ").strip()
            
            if not odia_input or odia_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            translation = translator.translate(odia_input)
            print(f"Translation: {translation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
