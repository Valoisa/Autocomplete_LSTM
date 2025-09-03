import string

class CharTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<eos>', '<unk>']
        self.alphabet = list(string.ascii_lowercase + string.digits + " .,!?;:'\"()-")
        self.vocab = self.special_tokens + self.alphabet
        
        self.char2id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}
        
        self.pad_token_id = self.char2id['<pad>']
        self.eos_token_id = self.char2id['<eos>']
        self.unk_token_id = self.char2id['<unk>']
        
    def encode(self, text: str, add_special_tokens=True):
        ids = [self.char2id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        chars = []
        if skip_special_tokens:
            chars = [
                self.id2char[i] 
                for i in ids 
                if i not in [self.pad_token_id, self.eos_token_id, self.unk_token_id]
            ]
        else:
            chars = [
                self.id2char[i] 
                for i in ids 
            ]
        
        return ''.join(chars)