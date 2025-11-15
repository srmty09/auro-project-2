#!/usr/bin/env python3
"""
Fast Gradio App for Odia-English Translation
Optimized for quick loading and responsive interface
"""

import gradio as gr
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import warnings
warnings.filterwarnings('ignore')

class FastTranslator:
    """Fast T5-based translator with lazy loading."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        print(f"🚀 Fast Translator initialized on {self.device}")
    
    def load_model_lazy(self):
        """Load model only when needed (lazy loading)."""
        if self.model_loaded:
            return True
            
        try:
            print("📦 Loading tokenizer...")
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            
            print("🤖 Loading model...")
            self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
            
            # Try to load trained weights
            try:
                print("⚡ Loading trained weights...")
                checkpoint = torch.load('best_translation_model.pt', map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    
                    # Map keys from t5_model.* format
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('t5_model.'):
                            new_key = key[9:]  # Remove 't5_model.'
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print("✅ Trained weights loaded!")
                else:
                    print("📝 Using base model")
                    
            except Exception as e:
                print(f"⚠️ Could not load trained weights: {e}")
                print("📝 Using base mT5 model")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def translate(self, odia_text, max_length=50):
        """Translate Odia text to English."""
        if not odia_text or not odia_text.strip():
            return "⚠️ Please enter some Odia text to translate."
        
        # Load model if not already loaded
        if not self.load_model_lazy():
            return "❌ Model failed to load. Please try again."
        
        try:
            input_text = f"translate Odia to English: {odia_text.strip()}"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=2,  # Reduced for speed
                    early_stopping=True,
                    do_sample=False
                )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.strip() if translation.strip() else "Translation not available"
            
        except Exception as e:
            return f"❌ Translation error: {str(e)}"

# Initialize translator (but don't load model yet)
translator = FastTranslator()

def translate_function(odia_text, max_length):
    """Gradio translation function."""
    return translator.translate(odia_text, max_length)

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Odia-English Translation",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown("""
        # 🇮🇳 Odia-English Translation System
        
        **Fast T5-based neural translation** - Enter Odia text and get English translation!
        
        *Note: Model loads on first translation (may take a moment)*
        """)
        
        with gr.Row():
            with gr.Column():
                odia_input = gr.Textbox(
                    label="Odia Text Input",
                    placeholder="Enter Odia text here... (e.g., ମୁଁ ଭଲ ଅଛି)",
                    lines=3
                )
                
                max_length = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Max Translation Length"
                )
                
                translate_btn = gr.Button("🔄 Translate", variant="primary")
                
            with gr.Column():
                output = gr.Textbox(
                    label="English Translation",
                    lines=3,
                    interactive=False
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["ମୁଁ ଭଲ ଅଛି", 30],
                ["ନମସ୍କାର", 20],
                ["ତୁମର ନାମ କଣ", 25],
            ],
            inputs=[odia_input, max_length],
            label="Try these examples:"
        )
        
        # Connect the translation function
        translate_btn.click(
            fn=translate_function,
            inputs=[odia_input, max_length],
            outputs=output
        )
        
        gr.Markdown("""
        ---
        **Status**: Model loads automatically on first use • **Architecture**: mT5-small with trained weights
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Fast Odia-English Translation App...")
    
    demo = create_interface()
    
    # Launch with auto-port finding
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,  # Auto-find available port
        share=False,
        show_error=True,
        quiet=False
    )
