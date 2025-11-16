import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from config import ImprovedConfig, DataConfig
from tokenizer import ImprovedTokenizer

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

HUGGINGFACE_DATASET = "OdiaGenAI/English_Odia_Parallel_Corpus"

class ImprovedOdiaEnglishDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: ImprovedTokenizer, config: ImprovedConfig, 
                 data_config: DataConfig, is_training: bool = True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        self.is_training = is_training
        
        self.data_pairs = self._load_data()
    
    def _load_data(self) -> List[Tuple[str, str]]:
        if DATASETS_AVAILABLE and self.data_path in ["./data/train.txt", "./data/val.txt", "./data/test.txt"]:
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
            return self._create_enhanced_sample_data()
    
    def _create_enhanced_sample_data(self) -> List[Tuple[str, str]]:
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
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [self.config.pad_token_id] * (max_length - len(seq))
        return seq
