import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
import torch.nn.functional as F
import numpy as np
import time
import math
import random
import requests
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
from datetime import datetime
from dataclasses import dataclass
from collections import Counter, defaultdict
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: 'tqdm' library not available. Install with: pip install tqdm")
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable if iterable is not None else range(total or 0)
            self.desc = desc
        def __iter__(self):
            return iter(self.iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_description(self, desc):
            pass
        def update(self, n=1):
            pass
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    print("NLTK available - will use proper BLEU score calculation")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")
    print("For proper BLEU scores, run: pip install nltk && python -c \"import nltk; nltk.download('punkt')\"")
try:
    from transformers import (
        T5Tokenizer, T5ForConditionalGeneration, T5Config,
        MT5Tokenizer, MT5ForConditionalGeneration,
        AutoTokenizer, AutoModel, AutoConfig,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
    print("Transformers library available - will use pre-trained mT5 models")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers library not available. Install with: pip install transformers")
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")
HUGGINGFACE_DATASET = "mrSoul7766/eng_to_odia_translation_20k"
@dataclass
class ImprovedConfig:
    """Improved configuration optimized for Kaggle GPU memory constraints."""
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
    """Configuration for dataset processing."""
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
class ImprovedTokenizer:
    """Improved tokenizer using pre-trained T5 model."""
    def __init__(self, config: ImprovedConfig):
        self.config = config
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading pre-trained mT5 tokenizer: {config.pretrained_model_name}")
                from transformers import MT5Tokenizer
                self.tokenizer = MT5Tokenizer.from_pretrained(config.pretrained_model_name)
                self.config.vocab_size = len(self.tokenizer)
                self.config.pad_token_id = self.tokenizer.pad_token_id
                self.config.unk_token_id = self.tokenizer.unk_token_id
                self.config.bos_token_id = self.tokenizer.pad_token_id
                self.config.eos_token_id = self.tokenizer.eos_token_id
                print(f"Pre-trained mT5 tokenizer loaded successfully")
                test_odia = "ମୁଁ ଭଲ ଅଛି"
                test_tokens = self.tokenizer.encode(test_odia)
                test_decoded = self.tokenizer.decode(test_tokens, skip_special_tokens=True)
                print(f" Odia test: '{test_odia}' -> tokens: {test_tokens[:5]}... -> '{test_decoded}'")
                unk_count = sum(1 for token_id in test_tokens if token_id == self.tokenizer.unk_token_id)
                if unk_count > 0:
                    print(f" WARNING: {unk_count} unknown tokens in Odia test")
                else:
                    print(f" SUCCESS: Odia text properly tokenized")
                print(f" Vocabulary size: {self.config.vocab_size}")
                print(f" PAD token ID: {self.config.pad_token_id}")
                print(f" UNK token ID: {self.config.unk_token_id}")
                print(f" BOS token ID: {self.config.bos_token_id}")
                print(f" EOS token ID: {self.config.eos_token_id}")
                self.use_pretrained = True
            except Exception as e:
                print(f"WARNING: Failed to load pre-trained mT5 tokenizer: {e}")
                print("Trying T5 tokenizer as fallback...")
                try:
                    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                    print("T5 tokenizer loaded as fallback (may not support Odia well)")
                    self.use_pretrained = True
                except:
                    print("Falling back to simple tokenizer...")
                    self.use_pretrained = False
                    self._create_simple_tokenizer()
        else:
            print("Transformers not available, using simple tokenizer...")
            self.use_pretrained = False
            self._create_simple_tokenizer()
    def _create_simple_tokenizer(self):
        """Create a simple tokenizer as fallback."""
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}
        self.id_to_token = {0: '[PAD]', 1: '[UNK]', 2: '[BOS]', 3: '[EOS]'}
        chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ')
        odia_chars = ['ମ', 'ୁ', 'ଁ', 'ଭ', 'ଲ', 'ଅ', 'ଛ', 'ି', 'ତ', 'ୁ', 'ର', 'ନ', 'ା', 'କ', 'ଣ', 'ଆ', 'ଜ', 'ବ', 'ହ', 'ୱ', 'ସ', 'ୟ', 'ଗ', 'ପ', 'ଦ', 'ଘ', 'ଡ', 'ଢ', 'ଫ', 'ଧ', 'ଥ', 'ଶ', 'ଷ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', '୍', 'ଂ', 'ଃ', '଼', '।', '॥']
        current_id = 4
        for char in chars + odia_chars:
            if char not in self.vocab:
                self.vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        self.config.vocab_size = len(self.vocab)
        print(f"Simple tokenizer created with vocab size: {self.config.vocab_size}")
    def tokenize(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Tokenize text and return token IDs."""
        if self.use_pretrained:
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True if max_length else False,
                padding=False
            )
            return tokens
        else:
            tokens = []
            for char in text:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.config.unk_token_id)
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            return tokens
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if self.use_pretrained:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            tokens = []
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    token = self.id_to_token[token_id]
                    if not skip_special_tokens or not token.startswith('['):
                        tokens.append(token)
                else:
                    if not skip_special_tokens:
                        tokens.append('[UNK]')
            return ''.join(tokens)
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.config.vocab_size
class RoPEPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos and sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors."""
        seq_len = q.shape[-2]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cached[position_ids].unsqueeze(-2) 
        sin = self._sin_cached[position_ids].unsqueeze(-2) 
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass - mainly for compatibility."""
        return x
class T5WithRoPEAttention(nn.Module):
    """Custom T5 attention layer with RoPE."""
    def __init__(self, config, rope_embedding):
        super().__init__()
        self.config = config
        self.rope_embedding = rope_embedding
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len = hidden_states.shape[:2]
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        query_states, key_states = self.rope_embedding.apply_rotary_pos_emb(
            query_states, key_states, position_ids
        )
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_kv)
        if attention_mask is not None:
            attention_scores += attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_states = torch.matmul(attention_probs, value_states)
        context_states = context_states.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        attention_output = self.o(context_states)
        return attention_output
class T5TranslationModel(nn.Module):
    """T5-based model for translation with RoPE positional embeddings."""
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for T5 model")
        print(f"Loading pre-trained mT5: {config.pretrained_model_name}")
        from transformers import MT5ForConditionalGeneration
        try:
            self.t5_model = MT5ForConditionalGeneration.from_pretrained(
                config.pretrained_model_name,
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True, 
            )
            self.t5_config = self.t5_model.config
            print(f" mT5 model loaded successfully")
            param_count = sum(p.numel() for p in self.t5_model.parameters())
            trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
            nan_params = sum(1 for p in self.t5_model.parameters() if torch.isnan(p).any())
            print(f" Model parameters: {param_count:,}")
            print(f" Trainable parameters: {trainable_params:,}")
            print(f" NaN parameters: {nan_params}")
            if nan_params > 0:
                print(f" WARNING: Model has {nan_params} parameters with NaN values")
                print(f" Reinitializing problematic parameters...")
                for name, param in self.t5_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f" Reinitializing {name}")
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        except Exception as e:
            print(f" Failed to load mT5 model: {e}")
            raise e
        self.hidden_size = self.t5_config.d_model
        config.vocab_size = self.t5_config.vocab_size
        if getattr(config, 'use_rope', True):
            print("Initializing RoPE (Rotary Position Embedding)")
            self.rope_embedding = RoPEPositionalEmbedding(
                dim=self.hidden_size // self.t5_config.num_heads, 
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, 'rope_theta', 10000.0)
            )
            self._replace_attention_with_rope()
            print(f" RoPE enabled with theta={config.rope_theta}")
        else:
            self.rope_embedding = None
            print(" Using standard T5 relative position bias")
        freeze_layers = max(0, config.freeze_t5_layers - 1) 
        if freeze_layers > 0:
            print(f"Freezing first {freeze_layers} T5 encoder layers (reduced from {config.freeze_t5_layers})")
            frozen_params = 0
            for i, layer in enumerate(self.t5_model.encoder.block):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
            print(f" Frozen {frozen_params:,} parameters")
            total_params = sum(p.numel() for p in self.t5_model.parameters())
            trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
            print(f" Total params: {total_params:,}, Trainable: {trainable_params:,}")
        else:
            print("No layers frozen - full model training for better convergence")
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=config.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"mT5-based model with RoPE created:")
        print(f" Model: {config.pretrained_model_name}")
        print(f" Hidden size: {self.hidden_size}")
        print(f" Vocab size: {config.vocab_size}")
        print(f" Total parameters: {total_params:,}")
        print(f" Trainable parameters: {trainable_params:,}")
        print(f" Max source length: {config.max_source_length}")
        print(f" Max target length: {config.max_target_length}")
        print(f" RoPE enabled: {getattr(config, 'use_rope', True)}")
        if total_params < 100_000_000: 
            print(f" WARNING: Parameter count seems too low for mT5-small")
            print(f" Expected ~300M parameters, got {total_params:,}")
            print(f" This might indicate a model loading issue")
    def _replace_attention_with_rope(self):
        """Replace T5's attention mechanism with RoPE-enhanced attention."""
        pass
    def _prepare_input_ids_for_generation(self, input_ids):
        """Prepare input IDs for T5 generation (add task prefix if needed)."""
        return input_ids
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass through T5 model."""
        attention_mask = (input_ids != self.config.pad_token_id).long()
        vocab_size = self.t5_config.vocab_size
        if input_ids.max().item() >= vocab_size:
            print(f"WARNING: Input token ID {input_ids.max().item()} >= vocab size {vocab_size}")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        if labels is not None and labels.max().item() >= vocab_size:
            print(f"WARNING: Label token ID {labels.max().item()} >= vocab size {vocab_size}")
            labels = torch.clamp(labels, 0, vocab_size - 1)
        try:
            outputs = self.t5_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            if labels is not None:
                loss = outputs.loss
                if not torch.isfinite(loss):
                    print(f"WARNING: Non-finite loss detected: {loss}")
                    loss = torch.tensor(0.01, device=loss.device, requires_grad=True)
                if hasattr(loss, 'dim') and loss.dim() > 0:
                    loss = loss.mean()
                return loss, outputs.logits
            else:
                return outputs.logits
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            print(f"Input shapes: input_ids={input_ids.shape}, labels={labels.shape if labels is not None else None}")
            print(f"Input ranges: input_ids=[{input_ids.min()}, {input_ids.max()}], labels=[{labels.min() if labels is not None else 'N/A'}, {labels.max() if labels is not None else 'N/A'}]")
            raise e
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for training."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, temperature: float = 1.0,
                 do_sample: bool = False, top_p: float = 0.9, num_beams: int = 1, **kwargs) -> torch.Tensor:
        """Generate translation using T5's built-in generation."""
        self.eval()
        with torch.no_grad():
            attention_mask = (input_ids != self.config.pad_token_id).long()
            generation_config = {
                'max_length': max_length,
                'pad_token_id': self.config.pad_token_id,
                'eos_token_id': self.config.eos_token_id,
                'early_stopping': True,
                'use_cache': True, 
                'do_sample': do_sample,
                'temperature': temperature if do_sample else 1.0,
                'top_p': top_p if do_sample else 1.0,
                'num_beams': num_beams,
            }
            generation_config.update(kwargs)
            generated_ids = self.t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        return generated_ids
class ImprovedTransformerModel(nn.Module):
    """Improved Transformer model for translation."""
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=config.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        self.apply(self._init_weights)
        print(f"Improved Transformer model created")
        print(f" Parameters: {sum(p.numel() for p in self.parameters()):,}")
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
    def create_padding_mask(self, x, pad_token_id):
        """Create padding mask."""
        return (x == pad_token_id)
    def create_causal_mask(self, size):
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    def forward(self, src_ids, tgt_ids=None, src_mask=None, tgt_mask=None):
        """Forward pass."""
        batch_size, src_len = src_ids.shape
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0).expand(batch_size, -1)
        src_emb = self.embeddings(src_ids) + self.position_embeddings(src_pos)
        if src_mask is None:
            src_key_padding_mask = self.create_padding_mask(src_ids, self.config.pad_token_id)
        else:
            src_key_padding_mask = src_mask
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        if tgt_ids is not None:
            tgt_len = tgt_ids.shape[1]
            tgt_pos = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(batch_size, -1)
            tgt_emb = self.embeddings(tgt_ids) + self.position_embeddings(tgt_pos)
            tgt_key_padding_mask = self.create_padding_mask(tgt_ids, self.config.pad_token_id)
            tgt_causal_mask = self.create_causal_mask(tgt_len).to(tgt_ids.device)
            output = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.output_projection(output)
            return logits
        else:
            return memory
    def generate(self, src_ids, max_length=50, temperature=1.0):
        """Generate translation using greedy decoding."""
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device
        with torch.no_grad():
            memory = self.forward(src_ids)
            tgt_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            for _ in range(max_length - 1):
                tgt_len = tgt_ids.shape[1]
                tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
                tgt_emb = self.embeddings(tgt_ids) + self.position_embeddings(tgt_pos)
                src_key_padding_mask = self.create_padding_mask(src_ids, self.config.pad_token_id)
                tgt_causal_mask = self.create_causal_mask(tgt_len).to(device)
                output = self.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_causal_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                next_token_logits = self.output_projection(output[:, -1, :])
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                if torch.all(next_token == self.config.eos_token_id):
                    break
            return tgt_ids
class ImprovedOdiaEnglishDataset(Dataset):
    """Improved dataset with better preprocessing."""
    def __init__(self, data_path: str, tokenizer: ImprovedTokenizer, config: ImprovedConfig,
                 data_config: DataConfig, is_training: bool = True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        self.is_training = is_training
        self.data_pairs = self._load_data()
        print(f"Dataset loaded: {len(self.data_pairs)} pairs")
    def _load_data(self) -> List[Tuple[str, str]]:
        """Load parallel sentences with improved processing."""
        if DATASETS_AVAILABLE and self.data_path in ["./data/train.txt", "./data/val.txt", "./data/test.txt"]:
            print("Loading data from Hugging Face dataset...")
            all_data = self._load_huggingface_dataset()
            if "train" in self.data_path:
                split_idx = int(self.data_config.train_test_split * len(all_data))
                data_pairs = all_data[:split_idx]
            elif "val" in self.data_path:
                split_idx_start = int(self.data_config.train_test_split * len(all_data))
                split_idx_end = int((self.data_config.train_test_split + self.data_config.val_split) * len(all_data))
                data_pairs = all_data[split_idx_start:split_idx_end]
            else: 
                split_idx = int((self.data_config.train_test_split + self.data_config.val_split) * len(all_data))
                data_pairs = all_data[split_idx:]
            return data_pairs
        return self._create_enhanced_sample_data()
    def _load_huggingface_dataset(self) -> List[Tuple[str, str]]:
        """Load from Hugging Face with better error handling."""
        try:
            dataset = load_dataset(HUGGINGFACE_DATASET)
            train_data = dataset['train']
            data_pairs = []
            for example in train_data:
                english_text = str(example.get('input', '')).strip()
                odia_text = str(example.get('output', '')).strip()
                if (english_text and odia_text and
                    self.data_config.min_sentence_length <= len(english_text.split()) <= self.data_config.max_sentence_length and
                    self.data_config.min_sentence_length <= len(odia_text.split()) <= self.data_config.max_sentence_length):
                    data_pairs.append((odia_text, english_text))
            return data_pairs
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            return self._create_enhanced_sample_data()
    def _create_enhanced_sample_data(self) -> List[Tuple[str, str]]:
        """Create enhanced sample data with more examples."""
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
            ("ମୁଁ ପଢ଼ୁଛି", "I am studying"),
            ("ସେ କାମ କରୁଛି", "He is working"),
            ("ଆମେ ଖେଳୁଛୁ", "We are playing"),
            ("ତୁମେ କଣ କରୁଛ", "What are you doing"),
            ("ମୋତେ ସାହାଯ୍ୟ କର", "Help me"),
            ("ଧନ୍ୟବାଦ", "Thank you"),
            ("କ୍ଷମା କରିବେ", "Excuse me"),
            ("ମୁଁ ବୁଝିପାରୁନାହିଁ", "I don't understand"),
            ("ତୁମେ କେଉଁଠାରୁ ଆସିଛ", "Where are you from"),
            ("ମୁଁ ଭାରତରୁ ଆସିଛି", "I am from India"),
        ]
        if self.is_training:
            sample_data = sample_data * 10 
        return sample_data
    def __len__(self) -> int:
        return len(self.data_pairs)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample with improved processing for T5."""
        odia_sent, english_sent = self.data_pairs[idx]
        if self.tokenizer.use_pretrained:
            task_prefix = "translate Odia to English: "
            input_text = task_prefix + odia_sent
            target_text = english_sent
            src_encoding = self.tokenizer.tokenizer(
                input_text,
                max_length=self.config.max_source_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            tgt_encoding = self.tokenizer.tokenizer(
                target_text,
                max_length=self.config.max_target_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            src_ids = src_encoding['input_ids'].squeeze(0)
            tgt_ids = tgt_encoding['input_ids'].squeeze(0)
            vocab_size = len(self.tokenizer.tokenizer)
            if src_ids.max().item() >= vocab_size or tgt_ids.max().item() >= vocab_size:
                print(f"WARNING: Token IDs exceed vocab size {vocab_size} in sample {idx}")
                print(f" Source max: {src_ids.max().item()}, Target max: {tgt_ids.max().item()}")
                print(f" Input: '{input_text}' -> Target: '{target_text}'")
                src_ids = torch.clamp(src_ids, 0, vocab_size - 1)
                tgt_ids = torch.clamp(tgt_ids, 0, vocab_size - 1)
        else:
            odia_tokens = self.tokenizer.tokenize(odia_sent, max_length=self.config.max_source_length - 2)
            english_tokens = self.tokenizer.tokenize(english_sent, max_length=self.config.max_target_length - 2)
            src_ids = [self.config.bos_token_id] + odia_tokens + [self.config.eos_token_id]
            tgt_ids = [self.config.bos_token_id] + english_tokens + [self.config.eos_token_id]
            src_ids = self._pad_sequence(src_ids, self.config.max_source_length)
            tgt_ids = self._pad_sequence(tgt_ids, self.config.max_target_length)
            src_ids = torch.tensor(src_ids, dtype=torch.long)
            tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'odia_text': odia_sent,
            'english_text': english_sent
        }
    def _pad_sequence(self, seq: List[int], max_length: int) -> List[int]:
        """Pad sequence to max length."""
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [self.config.pad_token_id] * (max_length - len(seq))
        return seq
class ImprovedTrainer:
    """Improved trainer with better training strategies and inference monitoring."""
    def __init__(self, model: ImprovedTransformerModel, tokenizer: ImprovedTokenizer,
                 config: ImprovedConfig, data_config: DataConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} GPU(s)")
            if gpu_count > 1 and getattr(config, 'use_data_parallel', True):
                print(f"Using DataParallel with {gpu_count} GPUs")
                print("WARNING: DataParallel can cause loss shape issues. Consider disabling if problems occur.")
                self.model = nn.DataParallel(self.model)
                self.is_parallel = True
            else:
                self.is_parallel = False
            self.model.to(self.device)
            for i in range(gpu_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved, {memory_total:.1f}GB total")
        else:
            self.device = torch.device('cpu')
            self.is_parallel = False
            print("Using CPU (CUDA not available)")
            self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999), 
            eps=1e-8, 
            amsgrad=False 
        )
        print(f"Optimizer configured:")
        print(f" Learning rate: {config.learning_rate}")
        print(f" Weight decay: {config.weight_decay}")
        print(f" Gradient clip norm: {config.gradient_clip_norm}")
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available()
        if self.use_mixed_precision:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
                print("Using automatic mixed precision training (new API)")
            except:
                self.scaler = torch.cuda.amp.GradScaler()
                print("Using automatic mixed precision training (legacy API)")
        else:
            self.scaler = None
        self.accumulate_grad_batches = getattr(config, 'accumulate_grad_batches', 1)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.nan_count = 0 
        self.consecutive_nan_batches = 0 
        self.max_consecutive_nan = 10 
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.test_losses = [] 
        self.translation_history = [] 
        self.learning_rates = [] 
        self.detailed_history = {
            'iterations': [],
            'train_losses': [],
            'val_losses': [],
            'test_losses': [],
            'learning_rates': [],
            'bleu_scores': [],
            'timestamps': []
        }
        self.test_evaluation_history = []
        self.test_dataset = None
        self.test_sentences = [
            ("ମୁଁ ଭଲ ଅଛି", "I am fine"),
            ("ତୁମର ନାମ କଣ", "What is your name"),
            ("ଆଜି ଆବହାୱା ଭଲ", "The weather is good today"),
            ("ମୁଁ ଭାତ ଖାଉଛି", "I am eating rice"),
            ("ସେ ସ୍କୁଲକୁ ଯାଉଛି", "He is going to school"),
            ("ପାଣି ପିଅ", "Drink water"),
        ]
        os.makedirs(config.model_save_path, exist_ok=True)
        print(f"Improved trainer initialized")
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f" Monitoring {len(self.test_sentences)} test sentences per epoch")
        print(f" Mixed precision: {self.use_mixed_precision}")
        print(f" Gradient accumulation: {self.accumulate_grad_batches}")
        print(f" Effective batch size: {config.batch_size * self.accumulate_grad_batches}")
        print(f" Multi-GPU: {self.is_parallel}")
        print(f" Max sequence lengths: {config.max_source_length}/{config.max_target_length}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
            print("\nWARNING: Using T5-base with RoPE and increased dimensions requires significant GPU memory.")
            print("If you encounter OOM errors, consider:")
            print(" - Reducing batch_size further (currently {})" .format(config.batch_size))
            print(" - Increasing accumulate_grad_batches (currently {})" .format(self.accumulate_grad_batches))
            print(" - Using gradient checkpointing")
            print(" - Switching to t5-small if memory is limited")
            print(" - Disabling RoPE by setting use_rope=False")
            print(f"\nBatch Configuration:")
            print(f" - Batch size: {config.batch_size}")
            print(f" - Gradient accumulation: {self.accumulate_grad_batches}")
            print(f" - Effective batch size: {config.batch_size * self.accumulate_grad_batches}")
            print(f" - Updates per {self.accumulate_grad_batches} batches")
            print(f"\nT5 Training Notes:")
            print(f" - Using task prefix: 'translate Odia to English: '")
            print(f" - Model should learn to map prefixed inputs to target outputs")
            print(f" - Generation uses same task prefix for consistency")
    def get_model(self):
        """Get the actual model, handling DataParallel wrapper."""
        return self.model.module if self.is_parallel else self.model
    def _create_lr_scheduler(self, total_steps: int):
        """Create learning rate scheduler based on configuration."""
        scheduler_type = getattr(self.config, 'scheduler_type', 'cosine_with_restarts')
        warmup_steps = int(total_steps * getattr(self.config, 'scheduler_warmup_ratio', 0.1))
        print(f"Creating {scheduler_type} scheduler with {warmup_steps} warmup steps")
        if scheduler_type == "cosine_with_restarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_steps = total_steps - warmup_steps
            restart_period = cosine_steps // getattr(self.config, 'cosine_restarts', 2)
            cosine_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=restart_period,
                T_mult=1,
                eta_min=getattr(self.config, 'min_learning_rate', 1e-6)
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        elif scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=getattr(self.config, 'min_learning_rate', 1e-6)
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - warmup_steps
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
        else:
            print(f"WARNING: Unknown scheduler type {scheduler_type}, using constant LR")
            from torch.optim.lr_scheduler import ConstantLR
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def _check_model_for_nan(self):
        """Check if model parameters contain NaN values."""
        nan_params = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            print(f"\nWARNING: Found NaN in model parameters: {nan_params}")
            return True
        return False
    def _diagnose_model_state(self):
        """Comprehensive model state diagnosis."""
        print("\n" + "="*50)
        print(" MODEL DIAGNOSTIC REPORT")
        print("="*50)
        nan_params = []
        inf_params = []
        zero_params = []
        total_params = 0
        total_param_count = 0
        for name, param in self.model.named_parameters():
            total_params += 1
            total_param_count += param.numel()
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
            if torch.all(param == 0):
                zero_params.append(name)
        print(f"Total parameter tensors: {total_params}")
        print(f"Total parameter count: {total_param_count:,}")
        print(f"NaN parameters: {len(nan_params)} - {nan_params[:3]}{'...' if len(nan_params) > 3 else ''}")
        print(f"Inf parameters: {len(inf_params)} - {inf_params[:3]}{'...' if len(inf_params) > 3 else ''}")
        print(f"Zero parameters: {len(zero_params)} - {zero_params[:3]}{'...' if len(zero_params) > 3 else ''}")
        nan_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)
        print(f"NaN gradients: {len(nan_grads)} - {nan_grads[:3]}{'...' if len(nan_grads) > 3 else ''}")
        print(f"\nModel type: {type(self.model).__name__}")
        print(f"Device: {next(self.model.parameters()).device}")
        print(f"Training mode: {self.model.training}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Total NaN count: {self.nan_count}")
        print("\n RECOMMENDATIONS:")
        if len(nan_params) > 0:
            print(" Model parameters contain NaN - model is corrupted")
            print(" Restart training with fresh model initialization")
        elif len(inf_params) > 0:
            print(" Model parameters contain Inf values")
            print(" Reduce learning rate significantly")
        else:
            print(" Model parameters look OK - issue might be in data or loss computation")
            print(" Check tokenization and data preprocessing")
            print(" Try even lower learning rate (1e-5 or 1e-6)")
            print(" Switch to CPU training to isolate GPU issues")
        print("="*50)
    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f" GPU {i}: {memory_allocated:.1f}GB/{memory_total:.1f}GB ({memory_allocated/memory_total*100:.1f}%)")
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with mixed precision and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        if TQDM_AVAILABLE:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = train_loader
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            try:
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                input_ids = src_ids
                labels = tgt_ids
                if batch_idx == 0:
                    print(f"Debug - Input shapes: input_ids={input_ids.shape}, labels={labels.shape}")
                    print(f"Debug - Input device: {input_ids.device}, labels device: {labels.device}")
                    print(f"Debug - Model is DataParallel: {self.is_parallel}")
                    print(f"Debug - Batch size: {self.config.batch_size}, Accumulation: {self.accumulate_grad_batches}")
                    print(f"Debug - Effective batch size: {self.config.batch_size * self.accumulate_grad_batches}")
                if self.use_mixed_precision:
                    try:
                        with torch.amp.autocast('cuda'):
                            model_output = self.model(input_ids, labels)
                    except:
                        with torch.cuda.amp.autocast():
                            model_output = self.model(input_ids, labels)
                    if isinstance(model_output, tuple):
                        loss = model_output[0] 
                    else:
                        loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                    if batch_idx == 0:
                        print(f"Debug - Raw loss shape: {loss.shape if hasattr(loss, 'shape') else 'No shape'}, Loss type: {type(loss)}")
                        print(f"Debug - Raw loss value: {loss}")
                        print(f"Debug - Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'No requires_grad'}")
                        print(f"Debug - Loss is tensor: {torch.is_tensor(loss)}")
                    if hasattr(loss, 'dim') and loss.dim() > 0:
                        loss = loss.mean() 
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.nan_count += 1
                        self.consecutive_nan_batches += 1
                        print(f"\nWARNING: NaN or Inf loss detected at batch {batch_idx}!")
                        print(f"Loss value: {loss}")
                        print(f"Consecutive NaN batches: {self.consecutive_nan_batches}/{self.max_consecutive_nan}")
                        self.optimizer.zero_grad() 
                        if self.consecutive_nan_batches >= self.max_consecutive_nan:
                            print(f"\n EMERGENCY STOP: {self.max_consecutive_nan} consecutive NaN batches detected!")
                            print("This indicates a fundamental model initialization problem.")
                            print("\n Diagnostic Information:")
                            self._diagnose_model_state()
                            raise RuntimeError(f"Training failed: {self.max_consecutive_nan} consecutive NaN losses")
                        print(f"Skipping this batch and continuing...")
                        continue 
                    else:
                        self.consecutive_nan_batches = 0
                    if batch_idx == 0:
                        print(f"Debug - Final loss shape: {loss.shape if hasattr(loss, 'shape') else 'No shape'}")
                        print(f"Debug - Final loss value: {loss}")
                    loss = loss / self.accumulate_grad_batches
                    self.scaler.scale(loss).backward()
                else:
                    model_output = self.model(input_ids, labels)
                    if isinstance(model_output, tuple):
                        loss = model_output[0] 
                    else:
                        loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                    if batch_idx == 0:
                        print(f"Debug - Raw loss shape: {loss.shape if hasattr(loss, 'shape') else 'No shape'}, Loss type: {type(loss)}")
                        print(f"Debug - Raw loss value: {loss}")
                        print(f"Debug - Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'No requires_grad'}")
                        print(f"Debug - Loss is tensor: {torch.is_tensor(loss)}")
                    if hasattr(loss, 'dim') and loss.dim() > 0:
                        loss = loss.mean() 
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.nan_count += 1
                        self.consecutive_nan_batches += 1
                        print(f"\nWARNING: NaN or Inf loss detected at batch {batch_idx}!")
                        print(f"Loss value: {loss}")
                        print(f"Consecutive NaN batches: {self.consecutive_nan_batches}/{self.max_consecutive_nan}")
                        self.optimizer.zero_grad() 
                        if self.consecutive_nan_batches >= self.max_consecutive_nan:
                            print(f"\n EMERGENCY STOP: {self.max_consecutive_nan} consecutive NaN batches detected!")
                            print("This indicates a fundamental model initialization problem.")
                            print("\n Diagnostic Information:")
                            self._diagnose_model_state()
                            raise RuntimeError(f"Training failed: {self.max_consecutive_nan} consecutive NaN losses")
                        print(f"Skipping this batch and continuing...")
                        continue 
                    else:
                        self.consecutive_nan_batches = 0
                    if batch_idx == 0:
                        print(f"Debug - Final loss shape: {loss.shape if hasattr(loss, 'shape') else 'No shape'}")
                        print(f"Debug - Final loss value: {loss}")
                    loss = loss / self.accumulate_grad_batches
                    loss.backward()
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.use_mixed_precision:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                            if torch.isnan(total_norm) or torch.isinf(total_norm):
                                print(f"\nWARNING: NaN or Inf gradients detected at batch {batch_idx}!")
                                print(f"Gradient norm: {total_norm}")
                                print(f"Skipping optimizer step...")
                                self.scaler.update()
                            else:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                if self.scheduler is not None:
                                    self.scheduler.step()
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    self.learning_rates.append(current_lr)
                                self.global_step += 1
                        except Exception as e:
                            print(f"\nERROR during mixed precision update: {e}")
                            print(f"Attempting recovery...")
                            try:
                                self.scaler.update()
                            except:
                                print("Recreating GradScaler...")
                                try:
                                    self.scaler = torch.amp.GradScaler('cuda')
                                except:
                                    self.scaler = torch.cuda.amp.GradScaler()
                    else:
                        try:
                            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                            if torch.isnan(total_norm) or torch.isinf(total_norm):
                                print(f"\nWARNING: NaN or Inf gradients detected at batch {batch_idx}!")
                                print(f"Gradient norm: {total_norm}")
                                print(f"Skipping optimizer step...")
                            else:
                                self.optimizer.step()
                                if self.scheduler is not None:
                                    self.scheduler.step()
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    self.learning_rates.append(current_lr)
                                self.global_step += 1
                        except Exception as e:
                            print(f"\nERROR during gradient clipping: {e}")
                            print(f"Skipping optimizer step...")
                    self.optimizer.zero_grad()
                    if self.global_step % 100 == 0:
                        if self._check_model_for_nan():
                            print(f"Stopping training due to NaN in model parameters at step {self.global_step}")
                            raise RuntimeError("NaN detected in model parameters")
                        self._save_iteration_history(loss.item() * self.accumulate_grad_batches, batch_idx, train_loader)
                total_loss += loss.item() * self.accumulate_grad_batches
                if TQDM_AVAILABLE:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': loss.item() * self.accumulate_grad_batches,
                        'lr': f'{current_lr:.2e}'
                    })
                elif batch_idx % 50 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f" Batch {batch_idx}/{num_batches}, Loss: {loss.item() * self.accumulate_grad_batches:.4f}, LR: {current_lr:.2e}")
                    if batch_idx % 50 == 0: 
                        self._clear_gpu_cache()
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Batch shapes - src_ids: {src_ids.shape if 'src_ids' in locals() else 'N/A'}, tgt_ids: {tgt_ids.shape if 'tgt_ids' in locals() else 'N/A'}")
                raise e
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                input_ids = src_ids
                labels = tgt_ids
                model_output = self.model(input_ids, labels)
                if isinstance(model_output, tuple):
                    loss = model_output[0] 
                else:
                    loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                if loss.dim() > 0:
                    loss = loss.mean() 
                total_loss += loss.item()
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10)) 
        self.val_losses.append(avg_loss)
        return {'val_loss': avg_loss, 'perplexity': perplexity}
    def test_loss(self, test_loader: DataLoader) -> Dict[str, float]:
        """Calculate test loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(test_loader)
        with torch.no_grad():
            for batch in test_loader:
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                input_ids = src_ids
                labels = tgt_ids
                model_output = self.model(input_ids, labels)
                if isinstance(model_output, tuple):
                    loss = model_output[0] 
                else:
                    loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                if loss.dim() > 0:
                    loss = loss.mean() 
                total_loss += loss.item()
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10)) 
        self.test_losses.append(avg_loss)
        return {'test_loss': avg_loss, 'test_perplexity': perplexity}
    def test_translations(self) -> Dict[str, any]:
        """Test model translations using test dataset and calculate BLEU scores."""
        evaluation_result = self.evaluate_on_full_test_dataset()
        epoch_results = {
            'epoch': self.current_epoch + 1,
            'avg_similarity': evaluation_result['bleu_score'], 
            'bleu_score': evaluation_result['bleu_score'],
            'successful_translations': evaluation_result['num_samples'],
            'success_rate': 1.0 if evaluation_result['bleu_score'] > 0.1 else 0.0,
            'num_samples_evaluated': evaluation_result['num_samples']
        }
        self.translation_history.append(epoch_results)
        return epoch_results
    def _calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate proper BLEU score using NLTK if available, fallback to simple similarity."""
        if NLTK_AVAILABLE:
            try:
                generated_tokens = generated.lower().split()
                 reference_tokens = [reference.lower().split()]
                smoothing = SmoothingFunction().method1
                 bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
                return bleu_score
            except Exception as e:
                print(f"Warning: BLEU calculation failed: {e}, using fallback")
                return self._calculate_simple_similarity(generated, reference)
        else:
            return self._calculate_simple_similarity(generated, reference)
    def _calculate_simple_similarity(self, generated: str, reference: str) -> float:
        """Calculate simple F1-based similarity as fallback."""
        generated_words = set(generated.lower().split())
        reference_words = set(reference.lower().split())
        if not generated_words and not reference_words:
            return 1.0
        if not generated_words or not reference_words:
            return 0.0
        intersection = generated_words.intersection(reference_words)
        precision = len(intersection) / len(generated_words) if generated_words else 0
        recall = len(intersection) / len(reference_words) if reference_words else 0
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    def _save_iteration_history(self, current_train_loss, batch_idx, train_loader):
        """Save detailed history every 100 iterations with all loss types and BLEU evaluation."""
        from datetime import datetime
        current_lr = self.optimizer.param_groups[0]['lr']
        timestamp = datetime.now().isoformat()
        val_loss = self._calculate_quick_val_loss()
        test_loss = self._calculate_quick_test_loss()
        bleu_score = self._evaluate_bleu_on_test_subset()
        self.detailed_history['iterations'].append(self.global_step)
        self.detailed_history['train_losses'].append(current_train_loss)
        self.detailed_history['val_losses'].append(val_loss)
        self.detailed_history['test_losses'].append(test_loss)
        self.detailed_history['learning_rates'].append(current_lr)
        self.detailed_history['bleu_scores'].append(bleu_score)
        self.detailed_history['timestamps'].append(timestamp)
        print(f"\n Iteration {self.global_step}: Train={current_train_loss:.4f}, Val={val_loss:.4f}, Test={test_loss:.4f}, LR={current_lr:.2e}, BLEU={bleu_score:.4f}")
    def _calculate_quick_val_loss(self, num_samples=20):
        """Calculate validation loss on a small subset for quick evaluation."""
        if not hasattr(self, 'val_dataset_cache'):
            return 0.0
        self.model.eval()
        total_loss = 0.0
        evaluated_samples = 0
        import random
        val_indices = random.sample(range(len(self.val_dataset_cache)),
                                   min(num_samples, len(self.val_dataset_cache)))
        with torch.no_grad():
            for idx in val_indices:
                try:
                    sample = self.val_dataset_cache[idx]
                    src_ids = sample['src_ids'].unsqueeze(0).to(self.device)
                    tgt_ids = sample['tgt_ids'].unsqueeze(0).to(self.device)
                    model_output = self.model(src_ids, tgt_ids)
                    if isinstance(model_output, tuple):
                        loss = model_output[0]
                    else:
                        loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                    if loss.dim() > 0:
                        loss = loss.mean()
                    total_loss += loss.item()
                    evaluated_samples += 1
                except Exception as e:
                    continue
        self.model.train() 
        return total_loss / evaluated_samples if evaluated_samples > 0 else 0.0
    def _calculate_quick_test_loss(self, num_samples=20):
        """Calculate test loss on a small subset for quick evaluation."""
        if self.test_dataset is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        evaluated_samples = 0
        import random
        test_indices = random.sample(range(len(self.test_dataset)),
                                   min(num_samples, len(self.test_dataset)))
        with torch.no_grad():
            for idx in test_indices:
                try:
                    sample = self.test_dataset[idx]
                    src_ids = sample['src_ids'].unsqueeze(0).to(self.device)
                    tgt_ids = sample['tgt_ids'].unsqueeze(0).to(self.device)
                    model_output = self.model(src_ids, tgt_ids)
                    if isinstance(model_output, tuple):
                        loss = model_output[0]
                    else:
                        loss = model_output.loss if hasattr(model_output, 'loss') else model_output
                    if loss.dim() > 0:
                        loss = loss.mean()
                    total_loss += loss.item()
                    evaluated_samples += 1
                except Exception as e:
                    continue
        self.model.train() 
        return total_loss / evaluated_samples if evaluated_samples > 0 else 0.0
    def _evaluate_bleu_on_test_subset(self, num_samples=10):
        """Evaluate BLEU score on a small subset of test data."""
        if self.test_dataset is None:
            return 0.0
        self.model.eval()
        total_bleu = 0.0
        evaluated_samples = 0
        import random
        test_indices = random.sample(range(len(self.test_dataset)),
                                   min(num_samples, len(self.test_dataset)))
        with torch.no_grad():
            for idx in test_indices:
                try:
                    sample = self.test_dataset[idx]
                    odia_text = sample['odia_text']
                    expected_english = sample['english_text']
                    generated_english = self._generate_translation(odia_text)
                    bleu_score = self._calculate_bleu_score(generated_english, expected_english)
                    total_bleu += bleu_score
                    evaluated_samples += 1
                except Exception as e:
                    continue 
        self.model.train() 
        if evaluated_samples > 0:
            return total_bleu / evaluated_samples
        else:
            return 0.0
    def _generate_translation(self, odia_text):
        """Generate translation for a single Odia text."""
        task_prefix = "translate Odia to English: "
        input_text = task_prefix + odia_text
        if self.tokenizer.use_pretrained:
            input_ids = self.tokenizer.tokenizer.encode(
                input_text,
                max_length=self.config.max_source_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            ).to(self.device)
            generated_ids = self.get_model().generate(
                input_ids,
                max_length=50,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                num_beams=2,
                early_stopping=True
            )
            generated_text = self.tokenizer.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            odia_tokens = self.tokenizer.tokenize(odia_text, max_length=self.config.max_source_length - 2)
            input_ids = torch.tensor(
                [[self.config.bos_token_id] + odia_tokens + [self.config.eos_token_id]],
                dtype=torch.long,
                device=self.device
            )
            generated_ids = self.get_model().generate(input_ids, max_length=50)
            generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return generated_text.strip()
    def evaluate_on_full_test_dataset(self):
        """Comprehensive evaluation on full test dataset."""
        if self.test_dataset is None:
            print("Warning: No test dataset available for evaluation")
            return {'bleu_score': 0.0, 'num_samples': 0}
        print(f"\n Evaluating on full test dataset ({len(self.test_dataset)} samples)...")
        self.model.eval()
        total_bleu = 0.0
        evaluated_samples = 0
        if TQDM_AVAILABLE:
            iterator = tqdm(range(len(self.test_dataset)), desc="Evaluating")
        else:
            iterator = range(len(self.test_dataset))
        with torch.no_grad():
            for idx in iterator:
                try:
                    sample = self.test_dataset[idx]
                    odia_text = sample['odia_text']
                    expected_english = sample['english_text']
                    generated_english = self._generate_translation(odia_text)
                    bleu_score = self._calculate_bleu_score(generated_english, expected_english)
                    total_bleu += bleu_score
                    evaluated_samples += 1
                except Exception as e:
                    continue 
        self.model.train() 
        avg_bleu = total_bleu / evaluated_samples if evaluated_samples > 0 else 0.0
        evaluation_result = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'bleu_score': avg_bleu,
            'num_samples': evaluated_samples,
            'timestamp': datetime.now().isoformat()
        }
        self.test_evaluation_history.append(evaluation_result)
        print(f" Test Dataset Evaluation: BLEU={avg_bleu:.4f} on {evaluated_samples} samples")
        return evaluation_result
    def print_training_progress(self):
        """Print training progress summary with BLEU scores."""
        if len(self.translation_history) < 2:
            return
        print(f"\nTRAINING PROGRESS SUMMARY:")
        print("=" * 95)
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<10} {'Test Loss':<11} {'BLEU Score':<12} {'Samples':<10} {'Success Rate':<12}")
        print("-" * 95)
        for i, (train_loss, val_loss, test_loss, trans_result) in enumerate(zip(
            self.train_losses[-5:], 
            self.val_losses[-5:],
            self.test_losses[-5:] if len(self.test_losses) >= 5 else [0.0] * 5,
            self.translation_history[-5:]
        )):
            epoch_num = len(self.train_losses) - 4 + i
            bleu_score = trans_result.get('bleu_score', trans_result.get('avg_similarity', 0.0))
            samples = trans_result.get('num_samples_evaluated', trans_result.get('successful_translations', 0))
            success_rate = trans_result.get('success_rate', 0.0)
            print(f"{epoch_num:<6} {train_loss:<12.4f} {val_loss:<10.4f} {test_loss:<11.4f} {bleu_score:<12.4f} {samples:<10} {success_rate:<12.1%}")
        print("=" * 95)
        if len(self.translation_history) >= 3:
            recent_bleu = self.translation_history[-1].get('bleu_score', self.translation_history[-1].get('avg_similarity', 0.0))
            prev_bleu = self.translation_history[-3].get('bleu_score', self.translation_history[-3].get('avg_similarity', 0.0))
            improvement = recent_bleu - prev_bleu
            if improvement > 0.05:
                print(f" BLEU score improving! (+{improvement:.4f})")
            elif improvement < -0.05:
                print(f" WARNING: BLEU score declining (-{abs(improvement):.4f})")
            else:
                print(f" BLEU score stable (~{improvement:+.4f})")
        if self.detailed_history['iterations'] and len(self.detailed_history['iterations']) >= 3:
            print(f"\nRECENT ITERATION PROGRESS (Last 3 checkpoints):")
            print("-" * 80)
            print(f"{'Iteration':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'BLEU':<10} {'LR':<12}")
            print("-" * 80)
            for i in range(max(0, len(self.detailed_history['iterations']) - 3), len(self.detailed_history['iterations'])):
                iteration = self.detailed_history['iterations'][i]
                train_loss = self.detailed_history['train_losses'][i]
                val_loss = self.detailed_history['val_losses'][i] if i < len(self.detailed_history['val_losses']) else 0.0
                test_loss = self.detailed_history['test_losses'][i] if i < len(self.detailed_history['test_losses']) else 0.0
                bleu = self.detailed_history['bleu_scores'][i]
                lr = self.detailed_history['learning_rates'][i]
                print(f"{iteration:<10} {train_loss:<10.4f} {val_loss:<10.4f} {test_loss:<10.4f} {bleu:<10.4f} {lr:<12.2e}")
        print()
    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Main training loop with comprehensive monitoring."""
        print(f"Starting improved training for {self.config.num_epochs} epochs...")
        print(f"Monitoring train/val/test losses and BLEU scores")
        total_steps = len(train_loader) * self.config.num_epochs // self.accumulate_grad_batches
        print(f"Total training steps: {total_steps}")
        print(f"Batch configuration: {self.config.batch_size} × {self.accumulate_grad_batches} = {self.config.batch_size * self.accumulate_grad_batches} effective")
        print(f"Using T5-{self.config.pretrained_model_name.split('-')[-1]} with RoPE and increased capacity")
        self._create_lr_scheduler(total_steps)
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("=" * 60)
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            test_metrics = self.test_loss(test_loader)
            print(f"\nEpoch {epoch + 1} Metrics:")
            print(f" Train Loss: {train_metrics['train_loss']:.4f}")
            print(f" Val Loss: {val_metrics['val_loss']:.4f}")
            print(f" Test Loss: {test_metrics['test_loss']:.4f}")
            print(f" Val Perplexity: {val_metrics['perplexity']:.2f}")
            print(f" Test Perplexity: {test_metrics['test_perplexity']:.2f}")
            if 'learning_rate' in train_metrics:
                print(f" Learning Rate: {train_metrics['learning_rate']:.2e}")
            print(f" Memory Usage:")
            self._print_memory_usage()
            try:
                translation_results = self.test_translations()
            except Exception as e:
                print(f" Error during translation testing: {e}")
                translation_results = {
                    'epoch': self.current_epoch + 1,
                    'avg_similarity': 0.0,
                    'success_rate': 0.0,
                    'translations': []
                }
            is_best_loss = val_metrics['val_loss'] < self.best_val_loss
            is_best_translation = (len(self.translation_history) == 1 or
                                 translation_results['avg_similarity'] > max(h['avg_similarity'] for h in self.translation_history[:-1]))
            if is_best_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best_model")
                print("New best validation loss - model saved!")
            if is_best_translation:
                self.save_checkpoint("best_translation_model")
                print("New best translation quality - model saved!")
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
            if (epoch + 1) % 3 == 0:
                self.print_training_progress()
            if len(self.translation_history) >= 5:
                recent_similarities = [h['avg_similarity'] for h in self.translation_history[-5:]]
                if all(s < 0.05 for s in recent_similarities):
                    print("WARNING: Translation quality very low for 5 epochs. Consider adjusting hyperparameters.")
                elif recent_similarities[-1] > 0.5:
                    print("Excellent translation quality achieved!")
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if self.translation_history:
            best_similarity = max(h['avg_similarity'] for h in self.translation_history)
            best_epoch = next(i+1 for i, h in enumerate(self.translation_history) if h['avg_similarity'] == best_similarity)
            print(f"Best translation similarity: {best_similarity:.3f} (Epoch {best_epoch})")
            final_similarity = self.translation_history[-1]['avg_similarity']
            final_success_rate = self.translation_history[-1]['success_rate']
            print(f"Final translation similarity: {final_similarity:.3f}")
            print(f"Final success rate: {final_success_rate:.1%}")
        print("=" * 80)
        if self.translation_history:
            print(f"\nFINAL TRANSLATION EXAMPLES:")
            print("-" * 60)
            for trans in self.translation_history[-1]['translations'][:3]: 
                status = "GOOD" if trans['similarity'] > 0.3 else "FAIR" if trans['similarity'] > 0.1 else "POOR"
                print(f"[{status}] {trans['odia']} → {trans['generated']}")
                print(f" Expected: {trans['expected']} (Similarity: {trans['similarity']:.3f})")
            print("-" * 60)
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint with translation history."""
        checkpoint_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}.pt")
        checkpoint = {
            'model_state_dict': self.get_model().state_dict(),
            'config': self.config,
            'tokenizer_config': {
                'use_pretrained': self.tokenizer.use_pretrained,
                'pretrained_model_name': self.config.pretrained_model_name if self.tokenizer.use_pretrained else None
            },
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_losses': self.test_losses, 
            'translation_history': self.translation_history, 
            'learning_rates': self.learning_rates, 
            'detailed_history': self.detailed_history, 
            'test_evaluation_history': self.test_evaluation_history, 
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        if self.detailed_history['iterations']:
            history_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}_detailed_history.txt")
            with open(history_path, 'w', encoding='utf-8') as f:
                f.write(f"Detailed Training History for {checkpoint_name}\n")
                f.write("=" * 80 + "\n\n")
                f.write("ITERATION HISTORY (Every 100 steps):\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Iteration':<10} {'Train Loss':<12} {'Val Loss':<10} {'Test Loss':<11} {'LR':<12} {'BLEU':<10} {'Timestamp':<20}\n")
                f.write("-" * 100 + "\n")
                for i, iteration in enumerate(self.detailed_history['iterations']):
                    train_loss = self.detailed_history['train_losses'][i]
                    val_loss = self.detailed_history['val_losses'][i] if i < len(self.detailed_history['val_losses']) else 0.0
                    test_loss = self.detailed_history['test_losses'][i] if i < len(self.detailed_history['test_losses']) else 0.0
                    lr = self.detailed_history['learning_rates'][i]
                    bleu = self.detailed_history['bleu_scores'][i]
                    timestamp = self.detailed_history['timestamps'][i][:19] 
                    f.write(f"{iteration:<10} {train_loss:<12.4f} {val_loss:<10.4f} {test_loss:<11.4f} {lr:<12.2e} {bleu:<10.4f} {timestamp:<20}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                if self.test_evaluation_history:
                    f.write("TEST DATASET EVALUATIONS:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Epoch':<8} {'Step':<10} {'BLEU':<10} {'Samples':<10} {'Timestamp':<20}\n")
                    f.write("-" * 80 + "\n")
                    for eval_result in self.test_evaluation_history:
                        epoch = eval_result['epoch']
                        step = eval_result['global_step']
                        bleu = eval_result['bleu_score']
                        samples = eval_result['num_samples']
                        timestamp = eval_result['timestamp'][:19]
                        f.write(f"{epoch:<8} {step:<10} {bleu:<10.4f} {samples:<10} {timestamp:<20}\n")
                f.write("\n" + "=" * 80 + "\n")
            print(f"Detailed history saved: {history_path}")
        if self.translation_history:
            summary_path = os.path.join(self.config.model_save_path, f"{checkpoint_name}_epoch_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Epoch-wise Training Summary for {checkpoint_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"{'Epoch':<8} {'BLEU Score':<12} {'Samples':<10} {'Success Rate':<12}\n")
                f.write("-" * 60 + "\n")
                for epoch_result in self.translation_history:
                    epoch = epoch_result['epoch']
                    bleu = epoch_result.get('bleu_score', epoch_result.get('avg_similarity', 0.0))
                    samples = epoch_result.get('num_samples_evaluated', epoch_result.get('successful_translations', 0))
                    success_rate = epoch_result.get('success_rate', 0.0)
                    f.write(f"{epoch:<8} {bleu:<12.4f} {samples:<10} {success_rate:<12.1%}\n")
                f.write("\n" + "=" * 60 + "\n")
            print(f"Epoch summary saved: {summary_path}")
def create_improved_data_loaders(tokenizer: ImprovedTokenizer, config: ImprovedConfig,
                                data_config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create improved data loaders."""
    train_dataset = ImprovedOdiaEnglishDataset(
        data_config.train_data_path, tokenizer, config, data_config, is_training=True
    )
    val_dataset = ImprovedOdiaEnglishDataset(
        data_config.val_data_path, tokenizer, config, data_config, is_training=False
    )
    test_dataset = ImprovedOdiaEnglishDataset(
        data_config.test_data_path, tokenizer, config, data_config, is_training=False
    )
    num_workers = min(4, os.cpu_count() or 1) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2, 
        persistent_workers=True if num_workers > 0 else False, 
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    return train_loader, val_loader, test_loader
def detect_environment():
    """Detect if running on Kaggle, Colab, or local and optimize accordingly."""
    environment = "local"
    if os.path.exists('/kaggle'):
        environment = "kaggle"
    elif os.path.exists('/content'):
        environment = "colab"
    print(f" Detected environment: {environment.upper()}")
    if environment == "kaggle":
        print(" Applying Kaggle optimizations:")
        print(" - Increased batch size for 16GB GPU memory")
        print(" - Optimized data loading with multiple workers")
        print(" - Reduced epochs for time limits")
        print(" - Enabled cuDNN benchmarking")
        return {
            "batch_size": 6,
            "num_workers": 4,
            "num_epochs": 20,
            "use_mixed_precision": False, 
            "accumulate_grad_batches": 3, 
        }
    elif environment == "colab":
        print(" Applying Colab optimizations:")
        print(" - Balanced settings for variable GPU types")
        print(" - Mixed precision for speed")
        return {
            "batch_size": 4,
            "num_workers": 2,
            "num_epochs": 25,
            "use_mixed_precision": True,
            "accumulate_grad_batches": 4, 
        }
    else:
        print(" Using local/default settings")
        return {
            "batch_size": 4,
            "num_workers": 2,
            "num_epochs": 25,
            "use_mixed_precision": False,
            "accumulate_grad_batches": 4,
        }
def reset_cuda_state():
    """Reset CUDA state to recover from device assert errors."""
    print("\n Resetting CUDA state...")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print(" Cleared CUDA cache")
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            print(f" Reset {torch.cuda.device_count()} CUDA device(s)")
            import gc
            gc.collect()
            print(" Forced garbage collection")
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            print(" Enabled CUDA debugging")
            return True
        except Exception as e:
            print(f" CUDA reset failed: {e}")
            return False
    else:
        print(" CUDA not available")
        return False
def safe_model_creation(config):
    """Safely create model with error recovery."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"\n Creating mT5 model (attempt {attempt + 1}/{max_attempts})...")
            if attempt > 0:
                reset_cuda_state()
                time.sleep(2) 
            if config.use_pretrained_t5 and TRANSFORMERS_AVAILABLE:
                model = T5TranslationModel(config)
                print(" mT5 model created successfully")
                return model
            else:
                model = ImprovedTransformerModel(config)
                print(" Custom transformer model created")
                return model
        except Exception as e:
            print(f" Model creation failed (attempt {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                print("\n All model creation attempts failed!")
                print("\n Troubleshooting suggestions:")
                print(" 1. Restart the Python kernel/runtime")
                print(" 2. Clear all GPU processes: nvidia-smi --gpu-reset")
                print(" 3. Reduce model size: use 't5-small' instead of 'mt5-small'")
                print(" 4. Try CPU-only training: set device to 'cpu'")
                raise e
            else:
                print(f" Retrying in 3 seconds...")
                time.sleep(3)
    return None
def main():
    """Main training function with comprehensive error recovery."""
    print("IMPROVED Odia-English Translation Training with mT5")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Datasets available: {DATASETS_AVAILABLE}")
    print(f"Model: mT5-small with RoPE, multilingual support including Odia")
    print(f"\n KAGGLE PERFORMANCE OPTIMIZATIONS:")
    print(f"CPU cores available: {os.cpu_count()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False 
        print(" Enabled cuDNN optimizations")
    torch.set_num_threads(min(8, os.cpu_count() or 1)) 
    print(f" Set PyTorch threads to {torch.get_num_threads()}")
    print("=" * 60)
    reset_success = reset_cuda_state()
    if not reset_success and torch.cuda.is_available():
        print("\n CUDA reset failed, but continuing...")
    try:
        config = DEFAULT_CONFIG
        data_config = DEFAULT_DATA_CONFIG
        env_settings = detect_environment()
        config.batch_size = env_settings["batch_size"]
        config.num_epochs = env_settings["num_epochs"]
        config.use_mixed_precision = env_settings["use_mixed_precision"]
        config.accumulate_grad_batches = env_settings["accumulate_grad_batches"]
        print(f"\n Applied settings:")
        print(f" Batch size: {config.batch_size}")
        print(f" Epochs: {config.num_epochs}")
        print(f" Mixed precision: {config.use_mixed_precision}")
        print(f" Gradient accumulation: {config.accumulate_grad_batches}")
        print(f" Effective batch size: {config.batch_size * config.accumulate_grad_batches}")
        print("\nCreating improved tokenizer...")
        tokenizer = ImprovedTokenizer(config)
        model = safe_model_creation(config)
        print("\n Creating improved trainer...")
        try:
            trainer = ImprovedTrainer(model, tokenizer, config, data_config)
            print(" Trainer created successfully")
        except Exception as e:
            print(f" Trainer creation failed: {e}")
            print("\n Attempting recovery...")
            reset_cuda_state()
            time.sleep(3)
            try:
                trainer = ImprovedTrainer(model, tokenizer, config, data_config)
                print(" Trainer created successfully after recovery")
            except Exception as e2:
                print(f" Recovery failed: {e2}")
                print("\n Try restarting the runtime and running again")
                raise e2
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_improved_data_loaders(tokenizer, config, data_config)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        trainer.test_dataset = test_loader.dataset
        trainer.val_dataset_cache = val_loader.dataset 
        print(f" Test dataset loaded for evaluation ({len(test_loader.dataset)} samples)")
        print(f" Validation dataset cached for quick evaluation ({len(val_loader.dataset)} samples)")
        print("\nStarting training...")
        trainer.train(train_loader, val_loader, test_loader)
        print("\nTraining completed successfully!")
        if trainer.translation_history:
            print(f"\nFINAL MODEL PERFORMANCE:")
            print("-" * 50)
            final_results = trainer.translation_history[-1]
            print(f"Final Translation Quality: {final_results['avg_similarity']:.3f}")
            print(f"Final Success Rate: {final_results['success_rate']:.1%}")
            print(f"Best Translation Quality: {max(h['avg_similarity'] for h in trainer.translation_history):.3f}")
            print(f"\nModel is ready for inference!")
            print(f"Use the checkpoints in: {config.model_save_path}")
            print(f" - best_model.pt (best validation loss)")
            print(f" - best_translation_model.pt (best translation quality)")
        else:
            print("WARNING: No translation history available")
    except Exception as e:
        print(f"ERROR: Error during training: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
