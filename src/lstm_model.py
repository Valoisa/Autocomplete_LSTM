import torch
import torch.nn as nn

from torch.nn.functional import softmax


class LSTMAutoComplete(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.ffn = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.ffn.weight)
        nn.init.zeros_(self.ffn.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)

        logits = self.ffn(out)
        return logits
    
    def generate(self, input, eos_token, max_len=10, temperature=1.):
        self.eval()
        
        if isinstance(max_len, int):
            max_len = torch.full((input.size(0),), max_len, device=input.device)
        elif isinstance(max_len, list):
            max_len = torch.tensor(max_len, device=input.device)
        
        batch_size = input.size(0)
        completions = [[] for _ in range(batch_size)]
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=input.device)
        
        with torch.no_grad():
            current_input = input.clone()
            
            while active_mask.any() and current_input.size(1) < max_len.max().item():
                logits = self.forward(current_input)  # (batch_size, seq_len, vocab_size)
                
                next_token_logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
                probs = softmax(next_token_logits, dim=-1)
                
                next_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                for i in range(batch_size):
                    if active_mask[i]:
                        token = next_tokens[i].item()
                        
                        current_length = current_input.size(1)
                        if token == eos_token or current_length >= max_len[i].item():
                            active_mask[i] = False
                        else:
                            completions[i].append(token)
                
                if not active_mask.any():
                    break
                
                
                current_input = torch.cat([current_input, next_tokens], dim=1)
        
        return completions