import torch

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence


# Датасет для обучения next token prediction
class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        super().__init__()
        self.samples = []

        for text in texts:
            if text == '':
                continue
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            if (len(tokenized)) < seq_len:
                continue
            tokenized.append(tokenizer.eos_token_id)
            source = tokenized[:-1]
            target = tokenized[1:]
            self.samples.append((source, target))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        source, target = self.samples[idx]
        return {
            'source': torch.tensor(source), 
            'target': torch.tensor(target)
        }

 # Датасет для вычисления метрик ROUGE
 # 'input' содержит первые 3/4 текста, 'reference' содержит продолжение
class EvalROUGEDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        super().__init__()
        self.samples = []

        for text in texts:
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            if (len(tokenized)) < seq_len:
                continue
            max_len = len(tokenized)
            first_n = max_len // 4 * 3
            input = tokenized[:first_n]
            reference = tokenizer.decode(tokenized[first_n:])
            self.samples.append((input, reference, max_len))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input, reference, max_len = self.samples[idx]
        return {
            'input': torch.tensor(input), 
            'reference': reference,
            'max_len': max_len
        }
    
def collate_fn(batch):
    sources = [item['source'] for item in batch]
    targets = [item['target'] for item in batch]

    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    mask_sources = (padded_sources != 0).long()

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        'source': padded_sources,
        'mask': mask_sources,
        'target': padded_targets
    }
    
def collate_fn_rouge_ds(batch):
    inputs = [item['input'] for item in batch]
    references = [item['reference'] for item in batch]
    max_lens = [item['max_len'] for item in batch]

    padded_sources = pad_sequence(inputs, batch_first=True, padding_value=0)

    return {
        'input': padded_sources,
        'reference': references,
        'max_len': max_lens
    }
    