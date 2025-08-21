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
        completion = []
        with torch.no_grad():
            while input.shape[-1] < max_len:
                logits = self.forward(input)
                next_token_logits = logits[-1, :] / temperature
                probs = softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == eos_token:
                    break

                input = torch.cat([input, next_token], dim=-1)
                completion.append(next_token.item())
        return completion