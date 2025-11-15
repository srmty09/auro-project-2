import os
import re
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from config import BertConfig

class ManualTokenizer:
    """Manual tokenizer for Odia and English text processing."""
    
    def __init__(self, config: BertConfig):
        """
        Initialize the tokenizer.
        
        Args:
            config: BERT configuration containing vocab parameters
        """
        self.config = config
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': config.pad_token_id,
            '[UNK]': config.unk_token_id,
            '[CLS]': config.cls_token_id,
            '[SEP]': config.sep_token_id,
            '[MASK]': config.mask_token_id,
        }
        
        # Vocabulary dictionaries
        self.odia_vocab = {}
        self.english_vocab = {}
        self.odia_id_to_token = {}
        self.english_id_to_token = {}
        
        # Combined vocabulary for the model
        self.combined_vocab = {}
        self.id_to_token = {}
        
        # Initialize vocabularies
        self._initialize_vocabularies()
    
    def _initialize_vocabularies(self):
        """Initialize vocabularies with special tokens and basic tokens."""
        # Start with special tokens
        current_id = 0
        
        # Add special tokens to combined vocab
        for token, token_id in self.special_tokens.items():
            self.combined_vocab[token] = token_id
            self.id_to_token[token_id] = token
            current_id = max(current_id, token_id + 1)
        
        # Create basic Odia and English vocabularies
        self._create_basic_odia_vocab(current_id)
        self._create_basic_english_vocab(current_id + 1000)  # Offset for English
    
    def _create_basic_odia_vocab(self, start_id: int):
        """Create basic Odia vocabulary with common characters and words."""
        # Odia Unicode characters (basic set)
        odia_chars = [
            # Vowels
            'ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଏ', 'ଐ', 'ଓ', 'ଔ',
            # Consonants
            'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ',
            'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ',
            'ଟ', 'ଠ', 'ଡ', 'ଢ', 'ଣ',
            'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ',
            'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ',
            'ଯ', 'ର', 'ଲ', 'ଳ', 'ଵ', 'ଶ', 'ଷ', 'ସ', 'ହ',
            # Vowel signs
            'ା', 'ି', 'ୀ', 'ୁ', 'ୂ', 'ୃ', 'େ', 'ୈ', 'ୋ', 'ୌ', '୍',
            # Numbers
            '୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯'
        ]
        
        # Common Odia words
        common_odia_words = [
            'ମୁଁ', 'ତୁମେ', 'ସେ', 'ଆମେ', 'ତୁମମାନେ', 'ସେମାନେ',
            'ଅଛି', 'ଅଛୁ', 'ଅଛନ୍ତି', 'ଥିଲା', 'ଥିଲି', 'ଥିଲେ',
            'କରୁଛି', 'କରିବି', 'କରିଛି', 'କରିଥିଲି',
            'ଭଲ', 'ଖରାପ', 'ବଡ଼', 'ଛୋଟ', 'ନୂଆ', 'ପୁରୁଣା',
            'ଘର', 'ସ୍କୁଲ', 'କାମ', 'ଖାଦ୍ୟ', 'ପାଣି', 'ପୁସ୍ତକ',
            'ନାମ', 'ସମୟ', 'ଦିନ', 'ରାତି', 'ସକାଳ', 'ସନ୍ଧ୍ୟା',
            'କଣ', 'କିଏ', 'କେଉଁଠି', 'କେବେ', 'କିପରି', 'କାହିଁକି',
            'ଏବଂ', 'କିନ୍ତୁ', 'ଯଦି', 'ତେବେ', 'କିମ୍ବା', 'ନା'
        ]
        
        current_id = start_id
        
        # Add characters
        for char in odia_chars:
            if char not in self.combined_vocab:
                self.odia_vocab[char] = current_id
                self.combined_vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add common words
        for word in common_odia_words:
            if word not in self.combined_vocab:
                self.odia_vocab[word] = current_id
                self.combined_vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
    
    def _create_basic_english_vocab(self, start_id: int):
        """Create basic English vocabulary."""
        # English alphabet
        english_chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # Common English words
        common_english_words = [
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'good', 'bad', 'big', 'small', 'new', 'old', 'young',
            'house', 'school', 'work', 'food', 'water', 'book',
            'name', 'time', 'day', 'night', 'morning', 'evening',
            'what', 'who', 'where', 'when', 'how', 'why',
            'and', 'but', 'if', 'then', 'or', 'not', 'no', 'yes',
            'hello', 'hi', 'bye', 'please', 'thank', 'sorry'
        ]
        
        current_id = start_id
        
        # Add characters
        for char in english_chars:
            if char not in self.combined_vocab:
                self.english_vocab[char] = current_id
                self.combined_vocab[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add common words
        for word in common_english_words:
            if word not in self.combined_vocab:
                self.english_vocab[word] = current_id
                self.combined_vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
    
    def _preprocess_text(self, text: str, is_odia: bool = True) -> str:
        """Preprocess text before tokenization."""
        # Convert to lowercase for English
        if not is_odia:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle punctuation
        text = re.sub(r'([.!?])', r' \1 ', text)
        text = re.sub(r'([,;:])', r' \1 ', text)
        
        return text
    
    def tokenize(self, text: str, is_odia: bool = True) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text to tokenize
            is_odia: Whether the text is in Odia (True) or English (False)
        
        Returns:
            List of tokens
        """
        text = self._preprocess_text(text, is_odia)
        
        if is_odia:
            return self._tokenize_odia(text)
        else:
            return self._tokenize_english(text)
    
    def _tokenize_odia(self, text: str) -> List[str]:
        """Tokenize Odia text."""
        tokens = []
        words = text.split()
        
        for word in words:
            # Check if word exists in vocabulary
            if word in self.odia_vocab:
                tokens.append(word)
            else:
                # Character-level tokenization for unknown words
                for char in word:
                    if char in self.odia_vocab:
                        tokens.append(char)
                    else:
                        tokens.append('[UNK]')
        
        return tokens
    
    def _tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text."""
        tokens = []
        words = text.split()
        
        for word in words:
            # Remove punctuation for word lookup
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.english_vocab:
                tokens.append(clean_word)
                # Add punctuation as separate tokens
                punct = re.findall(r'[^\w]', word)
                tokens.extend(punct)
            else:
                # Subword tokenization (simple character-level for unknown words)
                for char in word:
                    if char in self.english_vocab:
                        tokens.append(char)
                    elif char.isspace():
                        continue
                    else:
                        tokens.append('[UNK]')
        
        return [token for token in tokens if token.strip()]
    
    def convert_tokens_to_ids(self, tokens: List[str], is_odia: bool = True) -> List[int]:
        """Convert tokens to IDs."""
        ids = []
        for token in tokens:
            if token in self.combined_vocab:
                ids.append(self.combined_vocab[token])
            else:
                ids.append(self.config.unk_token_id)
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                tokens.append('[UNK]')
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = self.convert_ids_to_tokens(ids)
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Simple joining - can be improved with proper detokenization
        text = ' '.join(tokens)
        
        # Clean up spacing around punctuation
        text = re.sub(r' ([.!?,:;])', r'\1', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def save_vocab(self, vocab_dir: str):
        """Save vocabulary to files."""
        os.makedirs(vocab_dir, exist_ok=True)
        
        # Save combined vocabulary
        vocab_file = os.path.join(vocab_dir, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.combined_vocab, f, ensure_ascii=False, indent=2)
        
        # Save ID to token mapping
        id_to_token_file = os.path.join(vocab_dir, 'id_to_token.json')
        with open(id_to_token_file, 'w', encoding='utf-8') as f:
            # Convert int keys to strings for JSON serialization
            id_to_token_str = {str(k): v for k, v in self.id_to_token.items()}
            json.dump(id_to_token_str, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {vocab_dir}")
        print(f"Total vocabulary size: {len(self.combined_vocab)}")
    
    def load_vocab(self, vocab_dir: str):
        """Load vocabulary from files."""
        vocab_file = os.path.join(vocab_dir, 'vocab.json')
        id_to_token_file = os.path.join(vocab_dir, 'id_to_token.json')
        
        if os.path.exists(vocab_file) and os.path.exists(id_to_token_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.combined_vocab = json.load(f)
            
            with open(id_to_token_file, 'r', encoding='utf-8') as f:
                id_to_token_str = json.load(f)
                # Convert string keys back to integers
                self.id_to_token = {int(k): v for k, v in id_to_token_str.items()}
            
            print(f"Vocabulary loaded from {vocab_dir}")
            print(f"Total vocabulary size: {len(self.combined_vocab)}")
        else:
            print(f"Vocabulary files not found in {vocab_dir}. Using default vocabulary.")
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.combined_vocab)
    
    def expand_vocab_from_corpus(self, corpus_file: str, max_vocab_size: int = 30000):
        """Expand vocabulary by analyzing a corpus file."""
        if not os.path.exists(corpus_file):
            print(f"Corpus file {corpus_file} not found. Skipping vocabulary expansion.")
            return
        
        print(f"Expanding vocabulary from {corpus_file}...")
        
        word_counts = Counter()
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    odia_sent, english_sent = line.split('\t', 1)
                    
                    # Process Odia sentence
                    odia_tokens = self.tokenize(odia_sent, is_odia=True)
                    word_counts.update(odia_tokens)
                    
                    # Process English sentence
                    english_tokens = self.tokenize(english_sent, is_odia=False)
                    word_counts.update(english_tokens)
        
        # Add most frequent words to vocabulary
        current_vocab_size = len(self.combined_vocab)
        remaining_slots = max_vocab_size - current_vocab_size
        
        most_common = word_counts.most_common(remaining_slots)
        
        current_id = max(self.id_to_token.keys()) + 1 if self.id_to_token else 0
        
        for word, count in most_common:
            if word not in self.combined_vocab:
                self.combined_vocab[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        print(f"Added {len(most_common)} new words to vocabulary")
        print(f"Final vocabulary size: {len(self.combined_vocab)}")

def create_tokenizer(config: BertConfig, corpus_file: Optional[str] = None) -> ManualTokenizer:
    """Create and initialize a tokenizer."""
    tokenizer = ManualTokenizer(config)
    
    if corpus_file and os.path.exists(corpus_file):
        tokenizer.expand_vocab_from_corpus(corpus_file, config.vocab_size)
    
    return tokenizer
