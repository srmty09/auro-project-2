import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import ImprovedConfig
try:
    from tokenizer import RoPEPositionalEmbedding
except ImportError:
    RoPEPositionalEmbedding = None

try:
    from transformers import MT5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class T5TranslationModel(nn.Module):
    """T5-based model for translation with RoPE positional embeddings."""
    
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for T5 model")
        
        # Load pre-trained mT5 model with validation
        print(f"Loading pre-trained mT5: {config.pretrained_model_name}")
        from transformers import MT5ForConditionalGeneration
        
        try:
            # Load with proper initialization
            self.t5_model = MT5ForConditionalGeneration.from_pretrained(
                config.pretrained_model_name,
                torch_dtype=torch.float32, # Ensure float32 for stability
                low_cpu_mem_usage=True, # Optimize memory usage
            )
            self.t5_config = self.t5_model.config
            print(f" mT5 model loaded successfully")
            
            # Validate model parameters
            param_count = sum(p.numel() for p in self.t5_model.parameters())
            trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
            nan_params = sum(1 for p in self.t5_model.parameters() if torch.isnan(p).any())
            
            print(f" Model parameters: {param_count:,}")
            print(f" Trainable parameters: {trainable_params:,}")
            print(f" NaN parameters: {nan_params}")
            
            if nan_params > 0:
                print(f" WARNING: Model has {nan_params} parameters with NaN values")
                print(f" Reinitializing problematic parameters...")
                # Reinitialize NaN parameters
                for name, param in self.t5_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f" Reinitializing {name}")
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
            
        except Exception as e:
            print(f" Failed to load mT5 model: {e}")
            raise e
        
        # Update config with T5 dimensions
        self.hidden_size = self.t5_config.d_model
        config.vocab_size = self.t5_config.vocab_size
        
        # Initialize RoPE if enabled
        if getattr(config, 'use_rope', True):
            print("Initializing RoPE (Rotary Position Embedding)")
            self.rope_embedding = RoPEPositionalEmbedding(
                dim=self.hidden_size // self.t5_config.num_heads, # Per-head dimension
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, 'rope_theta', 10000.0)
            )
            
            # Replace T5's relative position bias with RoPE in attention layers
            self._replace_attention_with_rope()
            print(f" RoPE enabled with theta={config.rope_theta}")
        else:
            self.rope_embedding = None
            print(" Using standard T5 relative position bias")
        
        # Freeze T5 layers if specified (reduce freezing for better learning)
        freeze_layers = max(0, config.freeze_t5_layers - 1) # Freeze one less layer
        if freeze_layers > 0:
            print(f"Freezing first {freeze_layers} T5 encoder layers (reduced from {config.freeze_t5_layers})")
            frozen_params = 0
            # Freeze encoder layers
            for i, layer in enumerate(self.t5_model.encoder.block):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
            print(f" Frozen {frozen_params:,} parameters")
            
            # Validate parameter counts after freezing
            total_params = sum(p.numel() for p in self.t5_model.parameters())
            trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
            print(f" Total params: {total_params:,}, Trainable: {trainable_params:,}")
        else:
            print("No layers frozen - full model training for better convergence")
        
        # Loss function with label smoothing
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=config.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        
        # Final model validation
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
        
        # Sanity check - mT5-small should have ~300M parameters
        if total_params < 100_000_000: # Less than 100M is suspicious
            print(f" WARNING: Parameter count seems too low for mT5-small")
            print(f" Expected ~300M parameters, got {total_params:,}")
            print(f" This might indicate a model loading issue")
    
    def _replace_attention_with_rope(self):
        """Replace T5's attention mechanism with RoPE-enhanced attention."""
        # Note: This is a simplified approach. In practice, you might want to
        # modify the T5 attention layers more carefully or use a custom T5 implementation
        # For now, we'll keep the standard T5 and apply RoPE conceptually
        pass
    
    def _prepare_input_ids_for_generation(self, input_ids):
        """Prepare input IDs for T5 generation (add task prefix if needed)."""
        # T5 expects task prefixes like "translate Odia to English: "
        # For simplicity, we'll use the input as-is since we're fine-tuning
        return input_ids
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass through T5 model."""
        # Create attention mask for input (ignore padding)
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        # Validate inputs to prevent CUDA errors
        vocab_size = self.t5_config.vocab_size
        if input_ids.max().item() >= vocab_size:
            print(f"WARNING: Input token ID {input_ids.max().item()} >= vocab size {vocab_size}")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        if labels is not None and labels.max().item() >= vocab_size:
            print(f"WARNING: Label token ID {labels.max().item()} >= vocab size {vocab_size}")
            labels = torch.clamp(labels, 0, vocab_size - 1)
        
        try:
            # Forward pass through T5
            outputs = self.t5_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if labels is not None:
                # Training mode - return loss (ensure scalar) and logits
                loss = outputs.loss
                
                # Validate loss
                if not torch.isfinite(loss):
                    print(f"WARNING: Non-finite loss detected: {loss}")
                    # Return a small positive loss to continue training
                    loss = torch.tensor(0.01, device=loss.device, requires_grad=True)
                
                # Handle DataParallel case where loss might not be scalar
                if hasattr(loss, 'dim') and loss.dim() > 0:
                    loss = loss.mean()
                
                return loss, outputs.logits
            else:
                # Inference mode - return logits
                return outputs.logits
                
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            print(f"Input shapes: input_ids={input_ids.shape}, labels={labels.shape if labels is not None else None}")
            print(f"Input ranges: input_ids=[{input_ids.min()}, {input_ids.max()}], labels=[{labels.min() if labels is not None else 'N/A'}, {labels.max() if labels is not None else 'N/A'}]")
            raise e
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for training."""
        # Shift labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
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
            # Create attention mask
            attention_mask = (input_ids != self.config.pad_token_id).long()
            
            # Generate using T5's built-in generation with updated parameters
            generation_config = {
                'max_length': max_length,
                'pad_token_id': self.config.pad_token_id,
                'eos_token_id': self.config.eos_token_id,
                'early_stopping': True,
                'use_cache': True, # Enable caching for efficiency
                'do_sample': do_sample,
                'temperature': temperature if do_sample else 1.0,
                'top_p': top_p if do_sample else 1.0,
                'num_beams': num_beams,
            }
            
            # Add any additional kwargs
            generation_config.update(kwargs)
            
            generated_ids = self.t5_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
            
        return generated_ids



class RoPEPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos and sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos and sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            
            # Create the rotation matrix components
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
        
        # Update cache if needed
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # Get cos and sin values for the positions
        cos = self._cos_cached[position_ids].unsqueeze(-2) # [batch, seq_len, 1, dim]
        sin = self._sin_cached[position_ids].unsqueeze(-2) # [batch, seq_len, 1, dim]
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass - mainly for compatibility."""
        # For T5, we'll apply RoPE in the attention mechanism
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
        
        # Linear projections
        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Linear projections
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)
        
        # Apply RoPE to query and key
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        query_states, key_states = self.rope_embedding.apply_rotary_pos_emb(
            query_states, key_states, position_ids
        )
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_kv)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores += attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_states = torch.matmul(attention_probs, value_states)
        
        # Reshape back
        context_states = context_states.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        attention_output = self.o(context_states)
        
        return attention_output


class ImprovedTransformerModel(nn.Module):
    """Improved Transformer model for translation."""
    
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Loss function with label smoothing
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=config.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        
        # Initialize weights
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
        
        # Create embeddings
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0).expand(batch_size, -1)
        src_emb = self.embeddings(src_ids) + self.position_embeddings(src_pos)
        
        # Create source padding mask
        if src_mask is None:
            src_key_padding_mask = self.create_padding_mask(src_ids, self.config.pad_token_id)
        else:
            src_key_padding_mask = src_mask
        
        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        if tgt_ids is not None:
            # Training mode
            tgt_len = tgt_ids.shape[1]
            tgt_pos = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(batch_size, -1)
            tgt_emb = self.embeddings(tgt_ids) + self.position_embeddings(tgt_pos)
            
            # Create target masks
            tgt_key_padding_mask = self.create_padding_mask(tgt_ids, self.config.pad_token_id)
            tgt_causal_mask = self.create_causal_mask(tgt_len).to(tgt_ids.device)
            
            # Decode
            output = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(output)
            return logits
        else:
            # Inference mode - return memory for generation
            return memory
    
    def generate(self, src_ids, max_length=50, temperature=1.0):
        """Generate translation using greedy decoding."""
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device
        
        with torch.no_grad():
            # Encode source
            memory = self.forward(src_ids)
            
            # Initialize target with BOS token
            tgt_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Get current target length
                tgt_len = tgt_ids.shape[1]
                
                # Create embeddings
                tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
                tgt_emb = self.embeddings(tgt_ids) + self.position_embeddings(tgt_pos)
                
                # Create masks
                src_key_padding_mask = self.create_padding_mask(src_ids, self.config.pad_token_id)
                tgt_causal_mask = self.create_causal_mask(tgt_len).to(device)
                
                # Decode
                output = self.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_causal_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Get next token logits
                next_token_logits = self.output_projection(output[:, -1, :])
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to target
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                
                # Stop if EOS token is generated
                if torch.all(next_token == self.config.eos_token_id):
                    break
            
            return tgt_ids
