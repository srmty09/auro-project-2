import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BertModel, BertEmbeddings, BertEncoder
from config import BertConfig
from typing import Optional, Tuple

class BertForTranslation(nn.Module):
    """BERT model for sequence-to-sequence translation."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Encoder (uses existing BERT model)
        self.encoder = BertModel(config)
        
        # Decoder layers
        self.decoder_embeddings = BertEmbeddings(config)
        self.decoder_layers = nn.ModuleList([
            BertDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, 
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the input sequence.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Encoded representations [batch_size, seq_len, hidden_size]
        """
        encoder_outputs, _ = self.encoder(input_ids, token_type_ids)
        return encoder_outputs
    
    def decode(self, encoder_outputs: torch.Tensor, decoder_input_ids: torch.Tensor,
               encoder_attention_mask: Optional[torch.Tensor] = None,
               decoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the target sequence.
        
        Args:
            encoder_outputs: Encoded source representations [batch_size, src_len, hidden_size]
            decoder_input_ids: Decoder input token IDs [batch_size, tgt_len]
            encoder_attention_mask: Encoder attention mask [batch_size, src_len]
            decoder_attention_mask: Decoder attention mask [batch_size, tgt_len]
        
        Returns:
            Decoder outputs [batch_size, tgt_len, hidden_size]
        """
        # Get decoder embeddings
        decoder_embeddings = self.decoder_embeddings(decoder_input_ids)
        
        hidden_states = decoder_embeddings
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask
            )
        
        return hidden_states
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training and inference.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, tgt_len]
            decoder_input_ids: Decoder input IDs [batch_size, tgt_len]
        
        Returns:
            Tuple of (logits, loss)
        """
        # Encode input
        encoder_outputs = self.encode(input_ids, token_type_ids, attention_mask)
        
        # Prepare decoder inputs
        if decoder_input_ids is None and labels is not None:
            # Teacher forcing: use labels shifted right
            decoder_input_ids = self._shift_right(labels)
        
        if decoder_input_ids is None:
            # For inference, start with CLS token
            batch_size = input_ids.size(0)
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.config.cls_token_id, 
                dtype=torch.long, 
                device=input_ids.device
            )
        
        # Create decoder attention mask
        decoder_attention_mask = self._create_decoder_attention_mask(decoder_input_ids)
        
        # Decode
        decoder_outputs = self.decode(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_outputs)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input IDs to the right for teacher forcing."""
        # Replace -100 (ignore tokens) with pad_token_id for embedding layer
        clean_input_ids = input_ids.clone()
        clean_input_ids[clean_input_ids == -100] = self.config.pad_token_id
        
        shifted_input_ids = clean_input_ids.new_zeros(clean_input_ids.shape)
        shifted_input_ids[..., 1:] = clean_input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.config.cls_token_id
        return shifted_input_ids
    
    def _create_decoder_attention_mask(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Create causal attention mask for decoder."""
        batch_size, seq_len = decoder_input_ids.shape
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=decoder_input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create padding mask
        padding_mask = (decoder_input_ids != self.config.pad_token_id).unsqueeze(1)
        padding_mask = padding_mask.expand(-1, seq_len, -1)
        
        # Combine masks
        attention_mask = causal_mask * padding_mask
        
        return attention_mask
    
    def generate(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None, max_length: int = 128,
                 num_beams: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate translations using beam search or greedy decoding.
        
        Args:
            input_ids: Source input IDs [batch_size, src_len]
            token_type_ids: Token type IDs [batch_size, src_len]
            attention_mask: Attention mask [batch_size, src_len]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search (1 for greedy)
            temperature: Sampling temperature
        
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode input
            encoder_outputs = self.encode(input_ids, token_type_ids, attention_mask)
            
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Initialize decoder input with CLS token
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.config.cls_token_id, 
                dtype=torch.long, 
                device=device
            )
            
            # Generate tokens one by one
            for _ in range(max_length - 1):
                # Create attention mask
                decoder_attention_mask = self._create_decoder_attention_mask(decoder_input_ids)
                
                # Decode
                decoder_outputs = self.decode(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    encoder_attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask
                )
                
                # Get logits for next token
                next_token_logits = self.output_projection(decoder_outputs[:, -1, :])
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token (greedy for now)
                if num_beams == 1:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    # Simple beam search implementation
                    next_token = torch.topk(next_token_logits, num_beams, dim=-1)[1][:, 0:1]
                
                # Append to decoder input
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
                # Check for end of sequence
                if torch.all(next_token == self.config.sep_token_id):
                    break
            
            return decoder_input_ids

class BertDecoderLayer(nn.Module):
    """BERT decoder layer with cross-attention."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = BertSelfAttention(config)
        self.self_attention_output = BertSelfOutput(config)
        
        # Cross-attention
        self.cross_attention = BertCrossAttention(config)
        self.cross_attention_output = BertSelfOutput(config)
        
        # Feed-forward
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            hidden_states: Decoder hidden states [batch_size, tgt_len, hidden_size]
            encoder_hidden_states: Encoder hidden states [batch_size, src_len, hidden_size]
            encoder_attention_mask: Encoder attention mask [batch_size, src_len]
            decoder_attention_mask: Decoder attention mask [batch_size, tgt_len, tgt_len]
        
        Returns:
            Updated hidden states [batch_size, tgt_len, hidden_size]
        """
        # Self-attention
        self_attention_outputs = self.self_attention(hidden_states, decoder_attention_mask)
        hidden_states = self.self_attention_output(self_attention_outputs, hidden_states)
        
        # Cross-attention
        cross_attention_outputs = self.cross_attention(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = self.cross_attention_output(cross_attention_outputs, hidden_states)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        hidden_states = self.output(intermediate_output, hidden_states)
        
        return hidden_states

class BertSelfAttention(nn.Module):
    """Modified BERT self-attention with optional attention mask."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores += (1.0 - attention_mask.unsqueeze(1)) * -10000.0
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class BertCrossAttention(nn.Module):
    """Cross-attention layer for decoder."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-attention between decoder and encoder states.
        
        Args:
            hidden_states: Decoder states (queries) [batch_size, tgt_len, hidden_size]
            encoder_hidden_states: Encoder states (keys, values) [batch_size, src_len, hidden_size]
            encoder_attention_mask: Encoder attention mask [batch_size, src_len]
        
        Returns:
            Context vectors [batch_size, tgt_len, hidden_size]
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply encoder attention mask
        if encoder_attention_mask is not None:
            attention_scores += (1.0 - encoder_attention_mask.unsqueeze(1).unsqueeze(1)) * -10000.0
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

# Import missing classes from original model
from model import BertSelfOutput, BertIntermediate, BertOutput
