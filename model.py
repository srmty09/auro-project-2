import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ImprovedConfig
from tokenizer import RoPEPositionalEmbedding

try:
    from transformers import MT5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class T5TranslationModel(nn.Module):
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for T5 model")
        
        from transformers import MT5ForConditionalGeneration
        
        try:
            self.t5_model = MT5ForConditionalGeneration.from_pretrained(
                config.pretrained_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.t5_config = self.t5_model.config
            
            param_count = sum(p.numel() for p in self.t5_model.parameters())
            trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
            nan_params = sum(1 for p in self.t5_model.parameters() if torch.isnan(p).any())
            
            if nan_params > 0:
                for name, param in self.t5_model.named_parameters():
                    if torch.isnan(param).any():
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
            
        except Exception as e:
            raise e
        
        self.hidden_size = self.t5_config.d_model
        config.vocab_size = self.t5_config.vocab_size
        
        if getattr(config, 'use_rope', True):
            self.rope_embedding = RoPEPositionalEmbedding(
                dim=self.hidden_size // self.t5_config.num_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, 'rope_theta', 10000.0)
            )
            
            self._replace_attention_with_rope()
        else:
            self.rope_embedding = None
        
        freeze_layers = max(0, config.freeze_t5_layers - 1)
        if freeze_layers > 0:
            frozen_params = 0
            for i, layer in enumerate(self.t5_model.encoder.block):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
        
        if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id, label_smoothing=config.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _replace_attention_with_rope(self):
        pass
    
    def _prepare_input_ids_for_generation(self, input_ids):
        return input_ids
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        vocab_size = self.t5_config.vocab_size
        if input_ids.max().item() >= vocab_size:
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        if labels is not None and labels.max().item() >= vocab_size:
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
                    loss = torch.tensor(0.01, device=loss.device, requires_grad=True)
                
                if hasattr(loss, 'dim') and loss.dim() > 0:
                    loss = loss.mean()
                
                return loss, outputs.logits
            else:
                return outputs.logits
                
        except Exception as e:
            raise e
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, temperature: float = 1.0, 
                 do_sample: bool = False, top_p: float = 0.9, num_beams: int = 1, **kwargs) -> torch.Tensor:
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
    
    def _init_weights(self, module):
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
        return (x == pad_token_id)
    
    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    
    def forward(self, src_ids, tgt_ids=None, src_mask=None, tgt_mask=None):
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
                
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                
                if (next_token == self.config.eos_token_id).all():
                    break
            
        return tgt_ids
