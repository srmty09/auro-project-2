#!/usr/bin/env python3
"""
Instant Gradio App - Starts immediately, loads model on demand
"""

import gradio as gr
import threading
import time

class InstantTranslator:
    """Instant translator that starts immediately."""
    
    def __init__(self):
        self.model_status = "not_loaded"
        self.model = None
        self.tokenizer = None
        print("🚀 Instant Translator ready!")
    
    def load_model_background(self):
        """Load model in background thread."""
        try:
            self.model_status = "loading"
            print("📦 Background loading started...")
            
            import torch
            from transformers import MT5ForConditionalGeneration, MT5Tokenizer
            import warnings
            warnings.filterwarnings('ignore')
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load tokenizer
            self.tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            
            # Load model
            self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
            
            # Try to load trained weights
            try:
                checkpoint = torch.load('best_translation_model.pt', map_location=device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('t5_model.'):
                            new_key = key[9:]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print("✅ Trained weights loaded!")
                
            except Exception as e:
                print(f"⚠️ Using base model: {e}")
            
            self.model.to(device)
            self.model.eval()
            self.model_status = "ready"
            print("✅ Model ready for translation!")
            
        except Exception as e:
            self.model_status = "error"
            print(f"❌ Model loading failed: {e}")
    
    def translate(self, odia_text, max_length=50):
        """Translate with status checking."""
        if not odia_text or not odia_text.strip():
            return "⚠️ Please enter some Odia text to translate."
        
        if self.model_status == "not_loaded":
            # Start loading in background
            threading.Thread(target=self.load_model_background, daemon=True).start()
            return "🔄 Model is loading... Please wait and try again in a moment."
        
        elif self.model_status == "loading":
            return "🔄 Model is still loading... Please wait a moment and try again."
        
        elif self.model_status == "error":
            return "❌ Model failed to load. Please restart the application."
        
        elif self.model_status == "ready":
            try:
                import torch
                
                input_text = f"translate Odia to English: {odia_text.strip()}"
                
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False
                    )
                
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translation.strip() if translation.strip() else "Translation not available"
                
            except Exception as e:
                return f"❌ Translation error: {str(e)}"

# Initialize translator
translator = InstantTranslator()

def translate_function(odia_text, max_length):
    """Gradio translation function."""
    return translator.translate(odia_text, max_length)

def get_status():
    """Get current model status."""
    status_map = {
        "not_loaded": "🔴 Model not loaded",
        "loading": "🟡 Model loading...",
        "ready": "🟢 Model ready",
        "error": "🔴 Model error"
    }
    return status_map.get(translator.model_status, "🔴 Unknown status")

# Create Gradio interface
with gr.Blocks(title="Odia-English Translation", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🇮🇳 Odia-English Translation System
    
    **Instant-start T5 translation** - App starts immediately, model loads on first use!
    """)
    
    # Status display
    status_display = gr.Textbox(
        label="Model Status",
        value=get_status(),
        interactive=False
    )
    
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
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            
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
    
    # Connect functions
    translate_btn.click(
        fn=translate_function,
        inputs=[odia_input, max_length],
        outputs=output
    )
    
    refresh_btn.click(
        fn=get_status,
        outputs=status_display
    )
    
    gr.Markdown("""
    ---
    **Instructions**: 
    1. App starts instantly 
    2. Click "Translate" to trigger model loading (first time only)
    3. Wait for model to load, then translate again
    4. Use "Refresh Status" to check loading progress
    """)

if __name__ == "__main__":
    print("🚀 Starting Instant Odia-English Translation App...")
    print("✅ App will start immediately - model loads on first translation")
    
    # Launch immediately
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,  # Use specific port
        share=False,
        show_error=True,
        quiet=False
    )
