# Complete Guide: Odia-English Translation Model

This guide explains everything about the Odia-English translation model in simple English.

## ⚠️ Important Notice

**You must train the model before using it for translation.** This project does not include pre-trained weights. Follow the training instructions below to create your own model.

## What This Project Does

This project translates text from Odia (an Indian language) to English using artificial intelligence. It uses a pre-trained model called mT5 (multilingual T5) and fine-tunes it specifically for Odia-English translation.

**Note:** You need to train the model yourself using your own data before you can use it for translation.

## Project Structure

```
project/
├── COMPLETE_GUIDE.md           # This guide (you are here)
├── config.py                   # Model settings and parameters
├── tokenizer.py               # Text processing (converts words to numbers)
├── model.py                   # The AI model architecture
├── dataset.py                 # Data loading and preprocessing
├── improved_kaggle_train.py   # Main training script
├── inference.py               # Simple translation tool
├── translate_cli.py           # Command-line translation tool
├── plot_training_metrics.py   # Creates training progress charts
├── monitor_training.py        # Live training monitor
├── best_translation_model.pt  # Trained model weights (created after training)
├── t5_weights_only.pt         # Clean model weights (created after training)
└── requirements_minimal.txt   # Required Python packages
```

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements_minimal.txt
```

### 2. Train the Model First
**Important: You must train the model before using it for translation.**

```bash
# Train the model (this will take time)
python improved_kaggle_train.py
```

### 3. Test Translation (After Training)
```bash
# Simple translation
python -c "from inference import quick_translate; print(quick_translate('ମୁଁ ଭଲ ଅଛି'))"

# Command-line tool
python translate_cli.py "ମୁଁ ଭଲ ଅଛି"

# Interactive mode
python translate_cli.py --interactive
```

## How to Use the Model

**⚠️ Important: Train the model first using `python improved_kaggle_train.py`**

### Method 1: Python Code
```python
from inference import quick_translate

# Translate single text
result = quick_translate("ମୁଁ ଭଲ ଅଛି")
print(result)  # Output: "I'm good."

# For more control
from inference import OdiaTranslator
translator = OdiaTranslator()
translation = translator.translate("ଧନ୍ୟବାଦ")
print(translation)  # Output: "Thank you"
```

### Method 2: Command Line
**Note: Only works after training the model**
```bash
# Single translation
python translate_cli.py "ମୁଁ ଭଲ ଅଛି"

# Interactive mode (type and translate)
python translate_cli.py --interactive

# Translate from file
python translate_cli.py --file input.txt --output output.txt

# Test the model
python translate_cli.py --test
```

## How the Model Works

### 1. Architecture
- **Base Model**: Google's mT5-small (300M parameters)
- **Fine-tuning**: Trained specifically on Odia-English pairs
- **Input Format**: "translate Odia to English: [odia text]"
- **Output**: English translation

### 2. Key Components

**Tokenizer** (`tokenizer.py`):
- Converts Odia and English text into numbers
- Uses mT5's multilingual vocabulary
- Handles special Odia characters properly

**Model** (`model.py`):
- T5TranslationModel: Main translation model
- ImprovedTransformerModel: Alternative architecture
- Both support advanced features like RoPE (Rotary Position Embedding)

**Dataset** (`dataset.py`):
- Loads Odia-English sentence pairs
- Preprocesses text for training
- Handles data augmentation

**Configuration** (`config.py`):
- ImprovedConfig: All model settings
- DataConfig: Data processing settings
- Easy to modify for experiments

## Training the Model

### Prerequisites
- Python 3.7+
- PyTorch 2.0+
- Transformers 4.20+
- At least 8GB RAM (16GB recommended)
- GPU recommended but not required

### Step 1: Prepare Data
The training script can use:
- Hugging Face dataset: "OdiaGenAI/English_Odia_Parallel_Corpus"
- Your own data files (one sentence pair per line, tab-separated)
- Built-in sample data for testing

### Step 2: Run Training
```bash
python improved_kaggle_train.py
```

### Step 3: Monitor Progress
```bash
# In another terminal, monitor live progress
python monitor_training.py

# After training, create plots
python plot_training_metrics.py
```

### Training Process Explained

1. **Data Loading**: Loads Odia-English sentence pairs
2. **Tokenization**: Converts text to numbers the model understands
3. **Model Setup**: Initializes mT5 model with custom settings
4. **Training Loop**: 
   - Shows model Odia sentences
   - Model predicts English translations
   - Compares predictions with correct answers
   - Adjusts model to improve accuracy
5. **Validation**: Tests model on unseen data
6. **Saving**: Saves best model weights

### Training Settings (config.py)

**Model Architecture**:
- Hidden size: 768 (model complexity)
- Attention heads: 12 (parallel processing)
- Layers: 8 (model depth)
- Max sequence length: 128 tokens

**Training Parameters**:
- Learning rate: 1e-4 (how fast model learns)
- Batch size: 6 (sentences processed together)
- Epochs: 20 (complete passes through data)
- Warmup steps: 300 (gradual learning rate increase)

**Advanced Features**:
- Label smoothing: Prevents overconfidence
- Gradient clipping: Prevents unstable training
- Learning rate scheduling: Optimizes learning over time

## Understanding Training Output

### During Training You'll See:
```
Epoch 1/20
==========
Training: 100%|████████| 50/50 [02:30<00:00, 3.00s/batch]
Train Loss: 2.1234, Val Loss: 2.3456, Test Loss: 2.2345
BLEU Score: 0.1234 (translation quality)

Translation Examples:
[GOOD] ମୁଁ ଭଲ ଅଛି → I am fine
Expected: I am fine (Similarity: 0.85)
```

### Key Metrics:
- **Loss**: Lower is better (measures prediction errors)
- **BLEU Score**: 0-1 scale, higher is better (translation quality)
- **Similarity**: How close generated translation is to expected

### Files Created During Training:
```
improved_checkpoints/
├── best_model.pt                    # Best validation loss
├── best_translation_model.pt        # Best translation quality
├── best_model_detailed_history.txt  # Training progress log
└── best_model_epoch_summary.txt     # Epoch-wise summary
```

## Troubleshooting

### Common Issues:

**1. "Model not loaded properly"**
- Make sure you have trained the model first: `python improved_kaggle_train.py`
- Check if `t5_weights_only.pt` exists in the project directory
- Try: `python -c "from inference import OdiaTranslator; OdiaTranslator()"`

**2. "CUDA out of memory"**
- Reduce batch_size in config.py (try 2 or 4)
- Use CPU training: set `force_cpu_training = True`

**3. "Poor translation quality"**
- Model needs more training data
- Try increasing num_epochs in config.py
- Check if using correct tokenizer

**4. "Training very slow"**
- Use GPU if available
- Reduce max_source_length and max_target_length
- Increase batch_size if memory allows

### Performance Tips:

**For Better Translations**:
- Use more training data
- Train for more epochs
- Adjust learning rate (try 5e-5 or 2e-4)
- Use beam search during inference

**For Faster Training**:
- Use GPU
- Increase batch size
- Reduce sequence lengths
- Use mixed precision training

## Model Performance

### Expected Model Results (After Training):
- Performance depends on training data quality and quantity
- BLEU scores vary by sentence complexity
- Best performance on common phrases and simple sentences
- Results improve with more training data and longer training time

### Example Translations:
```
Input:  ମୁଁ ଭଲ ଅଛି
Output: I'm good.
Expected: I am fine

Input:  ଧନ୍ୟବାଦ  
Output: Thank you.
Expected: Thank you

Input:  ତୁମର ନାମ କଣ
Output: What is your name?
Expected: What is your name
```

## Advanced Usage

### Custom Training Data
Create a file with tab-separated Odia-English pairs:
```
ମୁଁ ଭଲ ଅଛି	I am fine
ତୁମର ନାମ କଣ	What is your name
```

### Modify Training Settings
Edit `config.py`:
```python
class ImprovedConfig:
    learning_rate: float = 1e-4      # Learning speed
    batch_size: int = 6              # Memory usage
    num_epochs: int = 20             # Training duration
    max_source_length: int = 128     # Max Odia sentence length
    max_target_length: int = 128     # Max English sentence length
```

### Custom Model Architecture
Edit `model.py` to:
- Change model size (hidden_size, num_layers)
- Add custom loss functions
- Implement different attention mechanisms

## Visualization and Monitoring

### Training Progress Plots
```bash
python plot_training_metrics.py
```
Creates:
- Loss curves (training, validation, test)
- BLEU score progress
- Learning rate schedule
- Overfitting analysis

### Live Training Monitor
```bash
python monitor_training.py
```
Shows real-time:
- Current losses
- Recent progress
- Training speed
- Memory usage

## File Formats and Data

### Model Files:
- `.pt` files: PyTorch model weights
- `best_translation_model.pt`: Complete checkpoint with training history
- `t5_weights_only.pt`: Clean model weights for inference

### Text Files:
- `*_detailed_history.txt`: Iteration-by-iteration training log
- `*_epoch_summary.txt`: Epoch-wise performance summary
- Training data: Tab-separated Odia-English pairs

### Configuration:
- All settings in `config.py`
- Easy to modify for experiments
- Separate configs for model and data
