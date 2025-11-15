import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random
import requests
import json
import gzip
from pathlib import Path
from config import DataConfig, BertConfig

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")

class OdiaEnglishDataset(Dataset):
    """Dataset class for Odia-English parallel corpus."""
    
    def __init__(self, data_path: str, tokenizer, config: DataConfig, bert_config: BertConfig, is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the parallel corpus file
            tokenizer: Tokenizer instance for text processing
            config: Data configuration
            bert_config: BERT model configuration
            is_training: Whether this is for training (enables data augmentation)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.bert_config = bert_config
        self.is_training = is_training
        
        # Load and process data
        self.data_pairs = self._load_data()
        
    def _load_data(self) -> List[Tuple[str, str]]:
        """Load parallel sentences from file."""
        data_pairs = []
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found. Creating sample data.")
            return self._create_sample_data()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Assume format: odia_sentence\t\tenglish_sentence
        for line in lines:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    odia_sent = parts[0].strip()
                    english_sent = parts[1].strip()
                    
                    # Filter by length
                    if (self.config.min_sentence_length <= len(odia_sent.split()) <= self.config.max_sentence_length and
                        self.config.min_sentence_length <= len(english_sent.split()) <= self.config.max_sentence_length):
                        data_pairs.append((odia_sent, english_sent))
        
        print(f"Loaded {len(data_pairs)} sentence pairs from {self.data_path}")
        return data_pairs
    
    def _create_sample_data(self) -> List[Tuple[str, str]]:
        """Create sample Odia-English data for demonstration."""
        sample_data = [
            ("ମୁଁ ଭଲ ଅଛି", "I am fine"),
            ("ତୁମର ନାମ କଣ", "What is your name"),
            ("ଆଜି ଆବହାୱା ଭଲ", "The weather is good today"),
            ("ମୁଁ ଭାତ ଖାଉଛି", "I am eating rice"),
            ("ସେ ସ୍କୁଲକୁ ଯାଉଛି", "He is going to school"),
            ("ଏହା ଏକ ଭଲ ପୁସ୍ତକ", "This is a good book"),
            ("ମୋର ଘର ବଡ଼", "My house is big"),
            ("ତୁମେ କେମିତି ଅଛ", "How are you"),
            ("ଆମେ ବନ୍ଧୁ", "We are friends"),
            ("ପାଣି ପିଅ", "Drink water"),
        ]
        print(f"Created {len(sample_data)} sample sentence pairs")
        return sample_data
    
    def _augment_sentence(self, sentence: str) -> str:
        """Simple data augmentation by word shuffling (for demonstration)."""
        if not self.config.use_data_augmentation or random.random() > self.config.augmentation_prob:
            return sentence
        
        words = sentence.split()
        if len(words) > 2:
            # Randomly shuffle middle words, keep first and last
            middle = words[1:-1]
            random.shuffle(middle)
            return ' '.join([words[0]] + middle + [words[-1]])
        return sentence
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        odia_sent, english_sent = self.data_pairs[idx]
        
        # Apply augmentation if training
        if self.is_training:
            odia_sent = self._augment_sentence(odia_sent)
            english_sent = self._augment_sentence(english_sent)
        
        # Tokenize sentences
        odia_tokens = self.tokenizer.tokenize(odia_sent, is_odia=True)
        english_tokens = self.tokenizer.tokenize(english_sent, is_odia=False)
        
        # Convert to IDs
        odia_ids = self.tokenizer.convert_tokens_to_ids(odia_tokens, is_odia=True)
        english_ids = self.tokenizer.convert_tokens_to_ids(english_tokens, is_odia=False)
        
        # Ensure all token IDs are valid
        vocab_size = self.tokenizer.get_vocab_size()
        odia_ids = [min(max(0, token_id), vocab_size - 1) for token_id in odia_ids]
        english_ids = [min(max(0, token_id), vocab_size - 1) for token_id in english_ids]
        
        # Truncate sequences if they're too long (leave space for special tokens)
        max_seq_len = self.bert_config.max_position_embeddings - 3  # Reserve space for [CLS] and 2x [SEP]
        max_odia_len = max_seq_len // 2
        max_english_len = max_seq_len - max_odia_len
        
        if len(odia_ids) > max_odia_len:
            odia_ids = odia_ids[:max_odia_len]
        if len(english_ids) > max_english_len:
            english_ids = english_ids[:max_english_len]
        
        # Add special tokens: [CLS] odia_tokens [SEP] english_tokens [SEP]
        input_ids = [self.bert_config.cls_token_id] + odia_ids + [self.bert_config.sep_token_id] + english_ids + [self.bert_config.sep_token_id]
        
        # Create token type IDs (0 for Odia, 1 for English)
        token_type_ids = [0] * (len(odia_ids) + 2) + [1] * (len(english_ids) + 1)
        
        # Final safety check - ensure we don't exceed max_position_embeddings
        max_len = self.bert_config.max_position_embeddings
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            token_type_ids = token_type_ids[:max_len]
        
        # Pad to max length
        padding_length = max_len - len(input_ids)
        input_ids += [self.bert_config.pad_token_id] * padding_length
        token_type_ids += [0] * padding_length
        
        # Create attention mask
        attention_mask = [1 if token_id != self.bert_config.pad_token_id else 0 for token_id in input_ids]
        
        # For translation, we need labels (shifted English tokens)
        labels = english_ids + [self.bert_config.sep_token_id]
        if len(labels) > self.bert_config.max_target_length:
            labels = labels[:self.bert_config.max_target_length]
        
        # Pad labels
        label_padding = self.bert_config.max_target_length - len(labels)
        labels += [-100] * label_padding  # -100 is ignored in loss calculation
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'odia_text': odia_sent,
            'english_text': english_sent
        }

def create_data_loaders(tokenizer, data_config: DataConfig, bert_config: BertConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Create datasets
    train_dataset = OdiaEnglishDataset(
        data_config.train_data_path, 
        tokenizer, 
        data_config, 
        bert_config, 
        is_training=True
    )
    
    val_dataset = OdiaEnglishDataset(
        data_config.val_data_path, 
        tokenizer, 
        data_config, 
        bert_config, 
        is_training=False
    )
    
    test_dataset = OdiaEnglishDataset(
        data_config.test_data_path, 
        tokenizer, 
        data_config, 
        bert_config, 
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bert_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=bert_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=bert_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """Custom collate function for batching."""
    # This is handled by the dataset __getitem__ method
    # but can be customized here if needed
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'odia_texts': [item['odia_text'] for item in batch],
        'english_texts': [item['english_text'] for item in batch]
    }

class HuggingFaceDatasetLoader:
    """Loader for Hugging Face Odia-English parallel corpus."""
    
    def __init__(self, data_dir: str = "./data/huggingface", dataset_name: str = "mrSoul7766/eng_to_odia_translation_20k"):
        """
        Initialize Hugging Face dataset loader.
        
        Args:
            data_dir: Directory to store downloaded data
            dataset_name: Name of the Hugging Face dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.dataset = None
    
    def load_dataset_from_hf(self) -> bool:
        """Load dataset from Hugging Face."""
        if not DATASETS_AVAILABLE:
            print("Error: 'datasets' library not available. Install with: pip install datasets")
            return False
        
        try:
            print(f"Loading dataset '{self.dataset_name}' from Hugging Face...")
            self.dataset = load_dataset(self.dataset_name)
            print(f"Dataset loaded successfully!")
            
            # Print dataset info
            print(f"Dataset splits: {list(self.dataset.keys())}")
            for split_name, split_data in self.dataset.items():
                print(f"  {split_name}: {len(split_data)} examples")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset '{self.dataset_name}': {e}")
            return False
    
    def create_parallel_files(self) -> bool:
        """Create parallel corpus files in tab-separated format from Hugging Face dataset."""
        if self.dataset is None:
            print("Dataset not loaded. Please call load_dataset_from_hf() first.")
            return False
        
        try:
            # Handle different possible split names
            available_splits = list(self.dataset.keys())
            print(f"Available splits: {available_splits}")
            
            # Map common split names
            split_mapping = {
                'train': ['train', 'training'],
                'validation': ['validation', 'valid', 'val', 'dev'],
                'test': ['test', 'testing']
            }
            
            created_files = 0
            
            for target_split, possible_names in split_mapping.items():
                source_split = None
                for name in possible_names:
                    if name in available_splits:
                        source_split = name
                        break
                
                if source_split is None:
                    print(f"No {target_split} split found, skipping...")
                    continue
                
                print(f"Processing {source_split} split -> {target_split}_parallel.txt...")
                
                output_file = self.data_dir / f"{target_split}_parallel.txt"
                split_data = self.dataset[source_split]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    count = 0
                    for example in split_data:
                        # Handle different possible column names
                        english_text = None
                        odia_text = None
                        
                        # Try different possible column names for English
                        for eng_col in ['english', 'en', 'source', 'input', 'text']:
                            if eng_col in example:
                                english_text = example[eng_col]
                                break
                        
                        # Try different possible column names for Odia
                        for odia_col in ['odia', 'or', 'target', 'output', 'translation']:
                            if odia_col in example:
                                odia_text = example[odia_col]
                                break
                        
                        # If we couldn't find the columns, try to infer from the first example
                        if english_text is None or odia_text is None:
                            cols = list(example.keys())
                            print(f"Available columns: {cols}")
                            if len(cols) >= 2:
                                english_text = example[cols[0]]  # Assume first column is English
                                odia_text = example[cols[1]]     # Assume second column is Odia
                        
                        if english_text and odia_text:
                            # Clean the text
                            english_text = str(english_text).strip()
                            odia_text = str(odia_text).strip()
                            
                            if english_text and odia_text:
                                f.write(f"{odia_text}\t{english_text}\n")
                                count += 1
                
                print(f"Created {output_file} with {count} sentence pairs.")
                created_files += 1
            
            if created_files == 0:
                print("No parallel files created. Check dataset format.")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error creating parallel files: {e}")
            return False
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about the processed dataset."""
        stats = {}
        
        for split in ['train', 'validation', 'test']:
            parallel_file = self.data_dir / f"{split}_parallel.txt"
            
            if parallel_file.exists():
                with open(parallel_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    stats[split] = len(lines)
            else:
                stats[split] = 0
        
        return stats
    
    def setup_dataset(self) -> bool:
        """Complete setup of Hugging Face dataset."""
        print(f"=== Setting up {self.dataset_name} Dataset ===")
        
        # Load dataset from Hugging Face
        if not self.load_dataset_from_hf():
            print("Failed to load dataset from Hugging Face.")
            return False
        
        # Create parallel files
        if not self.create_parallel_files():
            print("Failed to create parallel files.")
            return False
        
        # Show statistics
        stats = self.get_dataset_stats()
        print("\nDataset Statistics:")
        for split, count in stats.items():
            if count > 0:
                print(f"  {split.capitalize()}: {count:,} sentence pairs")
        
        total = sum(stats.values())
        print(f"  Total: {total:,} sentence pairs")
        
        print(f"\n{self.dataset_name} dataset setup completed successfully!")
        return True

def setup_huggingface_dataset(data_config: DataConfig, dataset_name: str = "mrSoul7766/eng_to_odia_translation_20k") -> bool:
    """Setup Hugging Face dataset and update data config paths."""
    loader = HuggingFaceDatasetLoader(dataset_name=dataset_name)
    
    if not loader.setup_dataset():
        return False
    
    # Update data config paths to use Hugging Face data
    hf_dir = Path("./data/huggingface")
    data_config.train_data_path = str(hf_dir / "train_parallel.txt")
    data_config.val_data_path = str(hf_dir / "validation_parallel.txt")
    data_config.test_data_path = str(hf_dir / "test_parallel.txt")
    
    print(f"Updated data configuration to use {dataset_name} dataset.")
    return True

def create_data_loaders_with_huggingface(tokenizer, data_config: DataConfig, bert_config: BertConfig, 
                                        use_huggingface: bool = True, dataset_name: str = "mrSoul7766/eng_to_odia_translation_20k") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with option to use Hugging Face dataset."""
    
    if use_huggingface:
        print(f"Setting up {dataset_name} dataset...")
        if setup_huggingface_dataset(data_config, dataset_name):
            print(f"Using {dataset_name} dataset for training.")
        else:
            print(f"Failed to setup {dataset_name} dataset, falling back to sample data.")
            use_huggingface = False
    
    if not use_huggingface:
        print("Using sample dataset for training.")
    
    # Create datasets with updated paths
    return create_data_loaders(tokenizer, data_config, bert_config)

# Keep backward compatibility
def create_data_loaders_with_ai4bharat(tokenizer, data_config: DataConfig, bert_config: BertConfig, 
                                      use_ai4bharat: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with Hugging Face dataset (backward compatibility)."""
    return create_data_loaders_with_huggingface(tokenizer, data_config, bert_config, use_ai4bharat)
