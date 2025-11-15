# Model Files Guide

This document explains how to obtain and use the pretrained model files for the Odia-English Translation System.

## 📁 Model Files Overview

Due to GitHub's file size limitations (100MB), the large model files are hosted separately:

### Available Models

1. **`best_translation_model.pt`** (1.2GB)
   - Main BERT-based translation model
   - Ready-to-use for inference
   - Contains full model state dict

2. **`best_model/`** (1.2GB)
   - PyTorch saved model directory format
   - Contains model weights split into multiple files
   - Alternative format for the same model

## 🚀 Quick Start

### Option 1: Use Download Script (Recommended)
```bash
python download_models.py
```

### Option 2: Manual Download
1. Go to [Releases](https://github.com/YOUR_USERNAME/odia-english-translation/releases)
2. Download the model files from the latest release
3. Place them in the project root directory

### Option 3: Alternative Hosting
If GitHub releases are not available, models can be downloaded from:
- **Google Drive**: [Link to be added]
- **Hugging Face Hub**: [Link to be added]
- **Dropbox**: [Link to be added]

## 🔧 Using the Models

### With main.py
```bash
# Make sure model file exists
python main.py --mode translate --checkpoint best_translation_model.pt --text "ମୁଁ ଭଲ ଅଛି"
```

### With custom inference scripts
```python
import torch
from translation_model import BertForTranslation

# Load the model
model = BertForTranslation(config)
checkpoint = torch.load('best_translation_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 📊 Model Information

| File | Size | Format | Description |
|------|------|--------|-------------|
| `best_translation_model.pt` | 1.2GB | PyTorch checkpoint | Complete model with training state |
| `best_model/` | 1.2GB | PyTorch saved model | Model weights in directory format |

## 🔍 Verification

After downloading, verify the models work:

```bash
# Test with the main model
python main.py --mode translate --checkpoint best_translation_model.pt --text "ନମସ୍କାର"

# Expected output: "Hello" or similar greeting
```

## 🛠️ Troubleshooting

### Model Not Found Error
```
FileNotFoundError: [Errno 2] No such file or directory: 'best_translation_model.pt'
```
**Solution**: Run `python download_models.py` to download the models.

### Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Use CPU inference: Add `--device cpu` to your command
- Reduce batch size in the config
- Use a smaller model variant (if available)

### Corrupted Model File
```
RuntimeError: Error loading model
```
**Solution**: 
- Re-download the model file
- Check file integrity using the download script
- Ensure the file wasn't corrupted during transfer

## 📈 Model Performance

- **Training Dataset**: 20,000 Odia-English sentence pairs
- **Architecture**: BERT encoder-decoder
- **Vocabulary Size**: 10,000 tokens
- **Training Time**: ~X hours on GPU
- **Best Validation Loss**: X.XXXX

## 🔄 Model Updates

When new model versions are released:

1. Check the [Releases page](https://github.com/YOUR_USERNAME/odia-english-translation/releases)
2. Download the latest model files
3. Replace the old model files
4. Update your inference scripts if needed

## 📝 Creating Your Own Models

To train your own models:

1. Use `improved_kaggle_train.py` for training
2. Models will be saved to `checkpoints/` directory
3. Use the best performing checkpoint for inference

```bash
python improved_kaggle_train.py --epochs 10 --batch_size 16
```

## 🤝 Contributing Models

If you've trained better models:

1. Test the model thoroughly
2. Document the training process
3. Create a pull request with model information
4. We'll add it to the releases if it improves performance
