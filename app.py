#!/usr/bin/env python3
"""
Hugging Face Spaces App for Odia-English Translation
A web interface for the BERT-based translation system
"""

import gradio as gr
import torch
import os
import sys
from pathlib import Path
import traceback

# Import T5 translation components
try:
    from simple_t5_inference import SimpleT5Translator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure simple_t5_inference.py is present")

class SpacesTranslationApp:
    """Hugging Face Spaces app for Odia-English translation."""
    
    def __init__(self):
        self.translator = None
        self.model_loaded = False
        self.error_message = None
        
        # Try to initialize the translator
        self.initialize_translator()
    
    def initialize_translator(self):
        """Initialize the translation system."""
        try:
            print("🚀 Initializing Odia-English Translation System...")
            
            # Check for model files
            model_files = [
                "best_translation_model.pt",
                "best_model/",
                "checkpoints/best_model.pt"
            ]
            
            model_path = None
            for path in model_files:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                self.error_message = """
                ⚠️ **Model files not found!**
                
                Please download the model files first:
                1. Run `python download_models.py` in the terminal
                2. Or upload model files to this Space
                3. Or use the model files from the repository
                
                Expected files: `best_translation_model.pt` or `best_model/`
                """
                print("❌ No model files found")
                return
            
            # Initialize T5 translator
            self.translator = SimpleT5Translator(model_path)
            print(f"✅ T5 Translator initialized with model: {model_path}")
            
            self.model_loaded = True
            print("✅ Translation system initialized successfully!")
            
        except Exception as e:
            self.error_message = f"""
            ❌ **Error initializing translation system:**
            
            ```
            {str(e)}
            ```
            
            **Troubleshooting:**
            1. Make sure model files are uploaded
            2. Check that all dependencies are installed
            3. Verify the model file format is correct
            """
            print(f"❌ Initialization error: {e}")
            traceback.print_exc()
    
    def translate_text(self, odia_text, max_length=50):
        """Translate Odia text to English."""
        
        if not odia_text or not odia_text.strip():
            return "⚠️ Please enter some Odia text to translate."
        
        if not self.model_loaded:
            if self.error_message:
                return self.error_message
            else:
                return "❌ Translation system not initialized. Please check the logs."
        
        try:
            # Use the T5 translator's translate method
            result = self.translator.translate(odia_text.strip(), max_length=max_length)
            return f"**English Translation:** {result}"
            
        except Exception as e:
            error_msg = f"""
            ❌ **Translation Error:**
            
            ```
            {str(e)}
            ```
            
            **Input:** {odia_text}
            
            Please try:
            1. Shorter text
            2. Check if the text is in Odia script
            3. Restart the space if issues persist
            """
            print(f"❌ Translation error: {e}")
            return error_msg
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.model_loaded:
            return "❌ Model not loaded"
        
        try:
            info = f"""
            ## 🤖 Model Information
            
            - **Architecture:** BERT Encoder-Decoder
            - **Task:** Odia → English Translation
            - **Vocabulary Size:** {self.translator.config.vocab_size if hasattr(self.translator, 'config') else 'Unknown'}
            - **Max Length:** {self.translator.config.max_position_embeddings if hasattr(self.translator, 'config') else 'Unknown'}
            - **Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}
            - **Status:** ✅ Ready for translation
            
            ### Sample Translations:
            - ମୁଁ ଭଲ ଅଛି → I am fine
            - ନମସ୍କାର → Hello/Namaste
            - ତୁମର ନାମ କଣ → What is your name
            """
            return info
            
        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"

# Initialize the app
app = SpacesTranslationApp()

# Create Gradio interface
def create_interface():
    """Create the Gradio web interface."""
    
    with gr.Blocks(
        title="Odia-English Translation",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🇮🇳 Odia-English Translation System
        
        A BERT-based neural machine translation system for translating Odia text to English.
        
        **Instructions:**
        1. Enter Odia text in the input box
        2. Adjust max length if needed (for longer outputs)
        3. Click "Translate" to get English translation
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                odia_input = gr.Textbox(
                    label="Odia Text Input",
                    placeholder="Enter Odia text here... (e.g., ମୁଁ ଭଲ ଅଛି)",
                    lines=3,
                    max_lines=5
                )
                
                max_length = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Max Translation Length"
                )
                
                translate_btn = gr.Button("🔄 Translate", variant="primary")
                
            with gr.Column(scale=2):
                output = gr.Markdown(
                    label="English Translation",
                    value="Translation will appear here..."
                )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["ମୁଁ ଭଲ ଅଛି", 30],
                ["ନମସ୍କାର", 20],
                ["ତୁମର ନାମ କଣ", 25],
                ["ଆଜି ପାଗ କେମିତି ଅଛି", 40],
                ["ମୁଁ ଭାତ ଖାଇବି", 30]
            ],
            inputs=[odia_input, max_length],
            label="Example Odia Phrases"
        )
        
        # Model information
        with gr.Accordion("📊 Model Information", open=False):
            model_info = gr.Markdown(app.get_model_info())
        
        # Set up the translation function
        translate_btn.click(
            fn=app.translate_text,
            inputs=[odia_input, max_length],
            outputs=output
        )
        
        # Footer
        gr.Markdown("""
        ---
        **About:** This translation system uses a custom BERT architecture trained on Odia-English parallel data.
        
        **Repository:** [GitHub](https://github.com/srmty09/auro-project-2) | **Model:** Custom BERT Encoder-Decoder
        """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    
    # Launch configuration - local testing
    demo.launch(
        server_name="127.0.0.1",  # Local only for testing
        server_port=None,         # Auto-find available port
        share=False,              # No public link for local testing
        show_error=True,          # Show errors in interface
        quiet=False               # Show startup logs
    )
