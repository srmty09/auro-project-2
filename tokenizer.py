import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from config import ImprovedConfig

try:
    from transformers import MT5Tokenizer, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ImprovedTokenizer:
    def __init__(self, config: ImprovedConfig):
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers import MT5Tokenizer
                self.tokenizer = MT5Tokenizer.from_pretrained(config.pretrained_model_name)
                
                self.config.vocab_size = len(self.tokenizer)
                
                self.config.pad_token_id = self.tokenizer.pad_token_id
                self.config.unk_token_id = self.tokenizer.unk_token_id
                self.config.bos_token_id = self.tokenizer.pad_token_id
                self.config.eos_token_id = self.tokenizer.eos_token_id
                
                test_odia = "ମୁଁ ଭଲ ଅଛି"
                test_tokens = self.tokenizer.encode(test_odia)
                test_decoded = self.tokenizer.decode(test_tokens, skip_special_tokens=True)
                
                unk_count = sum(1 for token_id in test_tokens if token_id == self.tokenizer.unk_token_id)
                
                self.use_pretrained = True
                
            except Exception as e:
                try:
                    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                    self.use_pretrained = True
                except:
                    self.use_pretrained = False
                    self._create_simple_tokenizer()
        else:
            self.use_pretrained = False
            self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
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
    
    def tokenize(self, text: str, max_length: Optional[int] = None) -> List[int]:
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
        return self.config.vocab_size

class RoPEPositionalEmbedding(nn.Module):
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
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        cos = self._cos_cached[position_ids].unsqueeze(-2)
        sin = self._sin_cached[position_ids].unsqueeze(-2)
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return x

class T5WithRoPEAttention(nn.Module):
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
        
        context_states = context_states.transpose(1, 2).contiguous()
        context_states = context_states.view(batch_size, seq_len, self.d_model)
        
        output = self.o(context_states)
        
        return output
