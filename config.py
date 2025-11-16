from dataclasses import dataclass
from typing import Optional

class ImprovedConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    
    max_source_length: int = 128
    max_target_length: int = 128
    
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    batch_size: int = 6
    num_epochs: int = 20
    warmup_steps: int = 300
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    scheduler_type: str = "cosine_with_restarts"
    cosine_restarts: int = 2
    scheduler_warmup_ratio: float = 0.1
    
    model_save_path: str = "./improved_checkpoints"
    save_every_n_steps: int = 500
    
    pretrained_model_name: str = "google/mt5-small"
    use_pretrained_t5: bool = True
    vocab_size: int = 250112
    
    emergency_fallback_cpu: bool = True
    emergency_fallback_model: str = "t5-small"
    force_cpu_training: bool = False
    
    freeze_t5_layers: int = 2
    t5_dropout: float = 0.1
    
    use_rope: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None
    
    use_mixed_precision: bool = False
    accumulate_grad_batches: int = 4
    label_smoothing: float = 0.1
    
    use_data_parallel: bool = True
    device_ids: list = None

@dataclass
class DataConfig:
    train_data_path: str = "./data/train.txt"
    val_data_path: str = "./data/val.txt"
    test_data_path: str = "./data/test.txt"
    
    min_sentence_length: int = 3
    max_sentence_length: int = 200
    train_test_split: float = 0.8
    val_split: float = 0.1
    
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.1

DEFAULT_CONFIG = ImprovedConfig()
DEFAULT_DATA_CONFIG = DataConfig()
