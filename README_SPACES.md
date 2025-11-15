---
title: Odia-English Translation
emoji: 🇮🇳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Odia-English Translation System 🇮🇳

A BERT-based neural machine translation system for translating between Odia and English languages.

## 🚀 Features

- **Custom BERT Architecture**: Built from scratch using PyTorch with encoder-decoder architecture
- **Manual Tokenizer**: Custom tokenizer supporting both Odia and English text processing
- **Web Interface**: Interactive Gradio interface for easy translation
- **Pretrained Models**: Ready-to-use models trained on Odia-English parallel data

## 🎯 How to Use

1. **Enter Odia text** in the input box (e.g., "ମୁଁ ଭଲ ଅଛି")
2. **Adjust max length** if you want longer translations
3. **Click "Translate"** to get the English translation
4. **Try examples** provided below the input box

## 📊 Model Details

- **Architecture**: BERT Encoder-Decoder
- **Training Data**: 20,000 Odia-English sentence pairs
- **Vocabulary Size**: 10,000 tokens
- **Max Sequence Length**: 256 tokens
- **Framework**: PyTorch

## 🔧 Local Setup

If you want to run this locally:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/odia-english-translation
cd odia-english-translation

# Install dependencies
pip install -r requirements.txt

# Download models (if not included)
python download_models.py

# Run the Gradio app
python app.py
```

## 📝 Example Translations

| Odia | English |
|------|---------|
| ମୁଁ ଭଲ ଅଛି | I am fine |
| ନମସ୍କାର | Hello/Namaste |
| ତୁମର ନାମ କଣ | What is your name |
| ଆଜି ପାଗ କେମିତି ଅଛି | How is the weather today |

## 🤝 Contributing

This is an open-source project. Contributions are welcome!

- **GitHub Repository**: [https://github.com/srmty09/auro-project-2](https://github.com/srmty09/auro-project-2)
- **Issues & Suggestions**: Please open an issue on GitHub
- **Model Improvements**: Share your trained models or training improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Trained on publicly available Odia-English parallel data
- Web interface powered by Gradio

---

**Note**: This model is for research and educational purposes. Translation quality may vary depending on the input text complexity and domain.
