import re
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class GPTTokenizer:
    def __init__(self):
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.pattern = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        self.special_tokens: Dict[str, int] = {}
        self.special_tokens_inverse: Dict[int, str] = {}
        
    def add_special_token(self, token: str) -> int:
        """Add a special token to the vocabulary"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        
        token_id = len(self.vocab) + len(self.special_tokens)
        self.special_tokens[token] = token_id
        self.special_tokens_inverse[token_id] = token
        return token_id

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Train the tokenizer on the given text"""
        assert vocab_size >= 256, "Vocabulary size must be at least 256"
        
        # Initialize vocabulary with byte-level tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
        # Pre-tokenize the text
        words = self._pre_tokenize(text)
        
        # Get initial frequencies
        frequencies = self._get_stats(words)
        
        # Perform merges until we reach desired vocab size
        while len(self.vocab) + len(self.special_tokens) < vocab_size:
            if not frequencies:
                break
                
            # Find most frequent pair
            best_pair = max(frequencies, key=frequencies.get)
            best_freq = frequencies[best_pair]
            
            if best_freq < 2:
                break
                
            # Create new token
            new_id = len(self.vocab)
            merged = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged
            
            # Store the merge
            self.merges[best_pair] = new_id
            
            # Update all words with this pair
            words = self._merge_words(words, best_pair, new_id)
            
            # Recompute frequencies
            frequencies = self._get_stats(words)
            
            if verbose:
                print(f"Merged {best_pair} into {new_id} (freq: {best_freq})")
        
        # Build inverse vocabulary for faster encoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def _pre_tokenize(self, text: str) -> List[List[int]]:
        """Split text into words and convert to byte IDs"""
        words = []
        for token in re.findall(self.pattern, text):
            # Convert to bytes then to list of byte IDs
            byte_ids = list(token.encode('utf-8'))
            words.append(byte_ids)
        return words
    
    def _get_stats(self, words: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Count frequency of adjacent pairs"""
        frequencies = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                frequencies[pair] += 1
        return frequencies
    
    def _merge_words(self, words: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        """Merge the given pair in all words"""
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text into tensor of token IDs"""
        # Handle special tokens first
        for token, token_id in sorted(self.special_tokens.items(), key=lambda x: -len(x[0])):
            if token in text:
                # This is simplified - real implementation would need to handle splits
                pass
        
        # Pre-tokenize
        words = self._pre_tokenize(text)
        tokens = []
        
        for word in words:
            # Convert to list of byte IDs
            current_word = word.copy()
            
            # Keep merging until no more merges possible
            while True:
                # Find all possible merges
                merges = []
                for i in range(len(current_word) - 1):
                    pair = (current_word[i], current_word[i+1])
                    if pair in self.merges:
                        merges.append((i, pair))
                
                if not merges:
                    break
                
                # Apply merges greedily (leftmost first)
                merges.sort()
                i, pair = merges[0]
                merged_id = self.merges[pair]
                current_word = current_word[:i] + [merged_id] + current_word[i+2:]
            
            tokens.extend(current_word)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode tensor of token IDs back to text"""
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
            
        bytes_list = []
        for token_id in token_ids.tolist():
            if token_id in self.special_tokens_inverse:
                # Handle special tokens
                bytes_list.append(self.special_tokens_inverse[token_id].encode('utf-8'))
            else:
                bytes_list.append(self.vocab[token_id])
        
        # Concatenate all bytes and decode
        return b''.join(bytes_list).decode('utf-8', errors='replace')
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        state = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'pattern': self.pattern.pattern
        }
        torch.save(state, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'GPTTokenizer':
        """Load tokenizer from file"""
        state = torch.load(filepath)
        tokenizer = cls()
        tokenizer.vocab = state['vocab']
        tokenizer.merges = state['merges']
        tokenizer.special_tokens = state['special_tokens']
        tokenizer.special_tokens_inverse = {v: k for k, v in state['special_tokens'].items()}
        tokenizer.pattern = re.compile(state['pattern'])
        tokenizer.inverse_vocab = {v: k for k, v in state['vocab'].items()}
        return tokenizer