from dataclasses import dataclass
from typing import Optional

@dataclass
class BertConfig:
    """Configuration class for BERT model used in Odia-English translation."""
    
    # Model architecture parameters
    vocab_size: int = 10000  # Combined vocab for Odia + English
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 256
    
    # Dropout parameters
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Translation specific parameters
    max_source_length: int = 128  # Max length for Odia input
    max_target_length: int = 128  # Max length for English output
    
    # Special tokens
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Model saving
    model_save_path: str = "./checkpoints"
    save_every_n_steps: int = 1000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.max_source_length <= self.max_position_embeddings, \
            "max_source_length cannot exceed max_position_embeddings"
        assert self.max_target_length <= self.max_position_embeddings, \
            "max_target_length cannot exceed max_position_embeddings"

@dataclass
class DataConfig:
    """Configuration for dataset processing."""
    
    # Dataset paths
    train_data_path: str = "./data/train.txt"
    val_data_path: str = "./data/val.txt"
    test_data_path: str = "./data/test.txt"
    
    # Tokenizer paths
    odia_vocab_path: str = "./vocab/odia_vocab.txt"
    english_vocab_path: str = "./vocab/english_vocab.txt"
    
    # Data processing
    min_sentence_length: int = 3
    max_sentence_length: int = 200
    train_test_split: float = 0.9
    val_split: float = 0.1
    
    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1

# Default configurations
DEFAULT_BERT_CONFIG = BertConfig()
DEFAULT_DATA_CONFIG = DataConfig()