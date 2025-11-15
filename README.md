# Odia-English Translation System using BERT

A complete BERT-based neural machine translation system for translating between Odia and English languages.

## Features

- **Custom BERT Architecture**: Built from scratch using PyTorch with encoder-decoder architecture
- **Manual Tokenizer**: Custom tokenizer supporting both Odia and English text processing
- **Comprehensive Training Pipeline**: Complete training loop with validation, checkpointing, and metrics tracking
- **Weight Management**: Automatic model saving/loading with checkpoint support
- **Interactive Translation**: Command-line interface for real-time translation
- **Batch Processing**: Support for translating multiple texts at once

## Project Structure

```
project/
├── model.py              # Base BERT model components
├── config.py             # Configuration classes for model and data
├── tokenizer.py          # Manual tokenizer for Odia-English text
├── dataset.py            # Dataset loading and processing
├── translation_model.py  # BERT-based translation model
├── train.py              # Training loop and trainer class
├── main.py               # Main script assembling all components
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/odia-english-translation.git
cd odia-english-translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained models:
```bash
python download_models.py
```

**Note**: The pretrained model files are large (>1GB) and excluded from the repository to keep it lightweight. You can:
- Use the download script to get pretrained models
- Train your own models using `improved_kaggle_train.py`
- Use the base mT5 model without additional training

## Usage

### 1. Training a New Model

#### Training with Sample Data (Quick Start)
```bash
python main.py --mode train
```

#### Training with mrSoul7766 Dataset (Recommended)
```bash
python main.py --mode train --use-ai4bharat
```

#### With custom configuration:
```bash
python main.py --mode train --config custom_config.json --corpus path/to/corpus.txt
```

#### Resume training from a checkpoint:
```bash
python main.py --mode train --resume checkpoints/checkpoint_step_1000.pt --use-ai4bharat
```

### 2. Translating Text

Translate a single text:
```bash
python main.py --mode translate --checkpoint checkpoints/best_model.pt --text "ମୁଁ ଭଲ ଅଛି"
```

### 3. Interactive Mode

Start interactive translation session:
```bash
python main.py --mode interactive --checkpoint checkpoints/best_model.pt
```

## Configuration

The system uses two main configuration classes:

### BertConfig
- `vocab_size`: Vocabulary size (default: 30000)
- `hidden_size`: Hidden dimension (default: 768)
- `num_hidden_layers`: Number of transformer layers (default: 12)
- `num_attention_heads`: Number of attention heads (default: 12)
- `max_position_embeddings`: Maximum sequence length (default: 512)
- `learning_rate`: Learning rate (default: 2e-5)
- `batch_size`: Training batch size (default: 16)
- `num_epochs`: Number of training epochs (default: 10)

### DataConfig
- `train_data_path`: Path to training data
- `val_data_path`: Path to validation data
- `test_data_path`: Path to test data
- `min_sentence_length`: Minimum sentence length (default: 3)
- `max_sentence_length`: Maximum sentence length (default: 200)

## Datasets

### mrSoul7766/eng_to_odia_translation_20k Dataset (Recommended)

The system now supports the **mrSoul7766/eng_to_odia_translation_20k** dataset from Hugging Face, which contains 20,000 high-quality English-to-Odia parallel sentence pairs.

#### Features:
- **Large Scale**: 20,000 parallel sentence pairs
- **High Quality**: Curated English-to-Odia translations
- **Hugging Face Integration**: Seamless download and processing
- **Automatic Setup**: Built-in loader handles everything
- **Ready to Use**: Pre-processed and cleaned data

#### Usage:
```bash
# Download and setup mrSoul7766 dataset
python demo.py --ai4bharat

# Train with mrSoul7766 dataset
python main.py --mode train --use-ai4bharat

# Or use the quick script
./run.sh train-ai4bharat
```

### Sample Dataset

For quick testing and development, the system includes a small sample dataset with 15 Odia-English sentence pairs.

### Custom Data Format

Training data should be in tab-separated format:
```
odia_sentence\tenglish_sentence
ମୁଁ ଭଲ ଅଛି	I am fine
ତୁମର ନାମ କଣ	What is your name
```

## Model Architecture

The system uses a BERT-based encoder-decoder architecture:

1. **Encoder**: Standard BERT encoder for processing source (Odia) text
2. **Decoder**: BERT-based decoder with cross-attention for generating target (English) text
3. **Attention Mechanisms**: 
   - Self-attention in both encoder and decoder
   - Cross-attention between decoder and encoder
4. **Output Layer**: Linear projection to vocabulary for token generation

## Training Process

1. **Data Loading**: Parallel corpus is loaded and preprocessed
2. **Tokenization**: Text is tokenized using the custom tokenizer
3. **Model Training**: 
   - Teacher forcing during training
   - Cross-entropy loss with label smoothing
   - AdamW optimizer with linear warmup and decay
   - Gradient clipping for stability
4. **Validation**: Regular validation with perplexity calculation
5. **Checkpointing**: Automatic saving of best models and training state

## Checkpoints and Weights

The system automatically saves:
- **Best Model**: `checkpoints/best_model.pt` (lowest validation loss)
- **Final Model**: `checkpoints/final_model.pt` (after training completion)
- **Periodic Checkpoints**: `checkpoints/checkpoint_step_N.pt` (every N steps)
- **Epoch Checkpoints**: `checkpoints/epoch_N.pt` (after each epoch)

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Training configuration
- Training metrics and history

## Sample Data

The system includes sample Odia-English sentence pairs for demonstration:
- Basic greetings and common phrases
- Simple sentence structures
- Everyday vocabulary

## Extending the System

### Adding New Languages
1. Extend the tokenizer to support the new language's script
2. Update vocabulary creation in `tokenizer.py`
3. Modify data loading in `dataset.py`

### Improving the Model
1. Experiment with different architectures in `translation_model.py`
2. Add attention visualization
3. Implement beam search for better generation
4. Add BLEU score evaluation

### Custom Training
1. Modify training parameters in `config.py`
2. Add custom loss functions in `train.py`
3. Implement learning rate scheduling strategies

## Performance Tips

1. **GPU Usage**: The system automatically uses CUDA if available
2. **Batch Size**: Adjust based on available memory
3. **Sequence Length**: Shorter sequences train faster
4. **Vocabulary Size**: Smaller vocabularies reduce memory usage
5. **Mixed Precision**: Can be added for faster training on modern GPUs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Ensure CUDA is available and being used
3. **Poor Translation Quality**: 
   - Increase training data
   - Train for more epochs
   - Adjust learning rate
4. **Tokenization Issues**: Check Unicode handling for Odia text

### Debug Mode

Enable verbose logging by modifying the print statements in the training loop.

## Contributing

To contribute to this project:
1. Add new features or improvements
2. Test with different language pairs
3. Optimize performance
4. Add evaluation metrics
5. Improve documentation

## License

This project is open source and available under the MIT License.
