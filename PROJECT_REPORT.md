# Odia-English Neural Machine Translation Project Report

## Executive Summary

This report presents the development and evaluation of a neural machine translation system for Odia-English language pairs. The project utilizes a fine-tuned mT5-small model to perform translation tasks, with comprehensive evaluation metrics and performance analysis.

**Key Results:**
- Average BLEU Score: 0.139
- Average Word Overlap: 0.380
- Translation Quality: Requires significant improvement
- Best performing sentence achieved 0.417 BLEU score

## 1. Project Overview

### 1.1 Objective
Develop an end-to-end neural machine translation system capable of translating text from Odia (an Indian language) to English using state-of-the-art transformer architecture.

### 1.2 Approach
- **Base Model**: Google's mT5-small (multilingual T5)
- **Fine-tuning Strategy**: Task-specific fine-tuning on Odia-English parallel corpus
- **Architecture**: Encoder-decoder transformer with attention mechanism
- **Training Data**: Odia-English sentence pairs with data augmentation

### 1.3 Technical Stack
- **Framework**: PyTorch 2.0+
- **Model Library**: Hugging Face Transformers
- **Tokenization**: MT5Tokenizer with multilingual support
- **Evaluation**: Custom BLEU score implementation
- **Visualization**: Matplotlib for performance plots

## 2. System Architecture

### 2.1 Model Components

```
Input: Odia Text → Tokenizer → Encoder → Decoder → Output: English Text
                     ↓
              Task Prefix: "translate Odia to English: "
```

**Core Components:**
- **T5TranslationModel**: Wrapper around MT5ForConditionalGeneration
- **ImprovedTokenizer**: Handles Odia and English text preprocessing
- **ImprovedConfig**: Centralized configuration management
- **Dataset Pipeline**: Efficient data loading and preprocessing

### 2.2 Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model Size | mT5-small | 300M parameters |
| Hidden Size | 768 | Model dimensionality |
| Attention Heads | 12 | Multi-head attention |
| Layers | 8 | Transformer layers |
| Max Sequence Length | 128 | Input/output length limit |
| Vocabulary Size | 250,112 | mT5 multilingual vocab |

### 2.3 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Initial learning rate |
| Batch Size | 6 | Training batch size |
| Epochs | 20 | Maximum training epochs |
| Optimizer | AdamW | Adaptive learning rate |
| Scheduler | Cosine with Restarts | Learning rate scheduling |
| Gradient Clipping | 1.0 | Gradient norm clipping |

## 3. Implementation Details

### 3.1 File Structure
```
project/
├── config.py                   # Configuration classes
├── tokenizer.py                # Text processing and tokenization
├── model.py                    # Model architecture definitions
├── dataset.py                  # Data loading and preprocessing
├── improved_kaggle_train.py    # Main training script
├── inference.py                # Translation inference engine
├── translate_cli.py            # Command-line interface
├── evaluate_translations.py    # Evaluation and metrics
├── plot_training_metrics.py    # Training visualization
├── monitor_training.py         # Live training monitoring
└── t5_weights_only.pt          # Trained model weights
```

### 3.2 Key Features

**Training Features:**
- Comprehensive logging and checkpointing
- Real-time translation quality monitoring
- Automatic best model selection
- Memory-efficient training with gradient accumulation
- Multi-GPU support with DataParallel

**Inference Features:**
- Simple Python API for integration
- Command-line interface for batch processing
- Interactive translation mode
- Automatic output file generation
- Error handling and recovery

**Evaluation Features:**
- BLEU score calculation (1-4 gram precision)
- Word overlap similarity metrics
- Comprehensive performance visualization
- Detailed analysis reports

## 4. Performance Evaluation

### 4.1 Test Dataset
The model was evaluated on 3 representative Odia-English sentence pairs covering different domains:
1. **Media/Entertainment**: Television awards ceremony viewership
2. **News/Politics**: Ukrainian-Russian conflict reporting  
3. **Labor/Business**: MLB union affiliation news

### 4.2 Evaluation Metrics

#### 4.2.1 BLEU Scores (Bilingual Evaluation Understudy)
BLEU measures n-gram precision between model output and reference translations.

| Sentence | BLEU Score | Quality Assessment |
|----------|------------|-------------------|
| Sentence 1 | 0.000 | Poor - Major semantic errors |
| Sentence 2 | 0.000 | Poor - Missing key context |
| Sentence 3 | 0.417 | Moderate - Acceptable quality |
| **Average** | **0.139** | **Poor Overall** |

#### 4.2.2 Word Overlap Analysis

| Sentence | Word Overlap | Shared Concepts |
|----------|--------------|-----------------|
| Sentence 1 | 0.333 | Numbers, basic structure |
| Sentence 2 | 0.286 | Military terms, success |
| Sentence 3 | 0.522 | Organization names, membership |
| **Average** | **0.380** | **Moderate Overlap** |

#### 4.2.3 N-gram Precision Analysis

| Metric | Sentence 1 | Sentence 2 | Sentence 3 | Average |
|--------|------------|------------|------------|---------|
| 1-gram Precision | 0.545 | 0.375 | 0.684 | 0.535 |
| 2-gram Precision | 0.300 | 0.200 | 0.500 | 0.333 |
| 3-gram Precision | 0.111 | 0.071 | 0.353 | 0.178 |
| 4-gram Precision | 0.000 | 0.000 | 0.250 | 0.083 |

### 4.3 Detailed Translation Analysis

#### 4.3.1 Sentence 1 Analysis
**Odia Input**: ଏହି ଟେଲିଭିଜନ ପୁରସ୍କାର ସମାରୋହ 2021 କାର୍ଯ୍ୟକ୍ରମ ତୁଳନାରେ ପ୍ରାୟ 1.5 ନିୟୁତ ଦର୍ଶକଙ୍କୁ ହରାଇଥିଲା ।

**Model Output**: "The telescope created nearly 1.5 million viewers in the 2021 program."

**Reference**: "The television awards ceremony lost roughly 1.5 million viewers compared to its 2021 program."

**Issues Identified:**
- Critical semantic error: "telescope" instead of "television"
- Incorrect verb: "created" instead of "lost"
- Missing context: "awards ceremony" not translated
- Structural problems in understanding the comparison

#### 4.3.2 Sentence 2 Analysis
**Odia Input**: ପୂର୍ବରେ ଋଷିଆ ସେନା ବିରୋଧରେ ୟୁକ୍ରେନୀୟ ସେନା ପକ୍ଷରୁ ଜାରି ହୋଇଥିବା ପ୍ରତିଆକ୍ରମଣରେ ନୂଆ ସଫଳତା ମିଳିଛି ।

**Model Output**: "There's a new success in the Russia military against the Russian force, said the U.S. government."

**Reference**: "Ukrainian forces are claiming new success in their counteroffensive against Russian forces in the east."

**Issues Identified:**
- Missing key entity: "Ukrainian forces" not identified
- Incorrect attribution: Added "U.S. government" not in original
- Lost context: "counteroffensive" concept missing
- Geographical reference "in the east" omitted

#### 4.3.3 Sentence 3 Analysis (Best Performance)
**Odia Input**: ଏମଏଲବି ଖେଳାଳି ସଂଘ ଶେଷରେ ଏଏଫଏଲ-ସିଆଇଓର ସଦସ୍ୟ ହେବ ଏବଂ ବିଭିନ୍ନ ଶିଳ୍ପର ଅନ୍ୟ ୫୭ଟି ସଂଘ ସହିତ ଯୋଡ଼ି ହେବ ।

**Model Output**: "The MLB Players Association will be a member of the AFL-CIO and another 75 other unions with other projects."

**Reference**: "The MLB Players Association will finally be a member of the AFL-CIO, affiliating with 57 other unions across industries."

**Strengths:**
- Correct entity recognition: "MLB Players Association", "AFL-CIO"
- Proper sentence structure and grammar
- Accurate core meaning preservation

**Minor Issues:**
- Number discrepancy: "75" vs "57"
- Missing adverb: "finally" not translated
- Imprecise phrasing: "other projects" vs "across industries"

## 5. Performance Visualization

### 5.1 Generated Plots

The evaluation system generated comprehensive visualizations:

1. **translation_evaluation.png**: 
   - BLEU scores by sentence
   - Word overlap comparison
   - N-gram precision analysis
   - Overall performance summary

2. **detailed_comparison.png**:
   - Side-by-side metric comparison
   - Combined scoring visualization
   - Performance trends across sentences

### 5.2 Key Insights from Visualizations

- **Sentence 3** significantly outperforms others (0.417 BLEU vs 0.000)
- **1-gram precision** generally higher than higher-order n-grams
- **Word overlap** shows moderate correlation with BLEU scores
- **Performance variance** indicates inconsistent translation quality

## 6. Strengths and Limitations

### 6.1 Strengths

**Technical Strengths:**
- Modular, well-structured codebase
- Comprehensive evaluation framework
- Real-time monitoring capabilities
- Flexible configuration system
- Multi-modal inference options

**Model Strengths:**
- Good at proper noun recognition (organizations, numbers)
- Maintains basic sentence structure
- Handles multilingual tokenization effectively
- Reasonable performance on domain-specific content

### 6.2 Limitations

**Model Limitations:**
- Poor average BLEU score (0.139)
- Inconsistent translation quality
- Semantic errors in complex sentences
- Limited contextual understanding
- Vocabulary gaps for specialized terms

**Technical Limitations:**
- Small model size (mT5-small) limits capacity
- Limited training data diversity
- No domain adaptation mechanisms
- Basic preprocessing pipeline
