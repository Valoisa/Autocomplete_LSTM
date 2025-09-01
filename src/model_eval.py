from tqdm.auto import tqdm

from transformers import BertTokenizerFast

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch

from typing import Optional

import evaluate

from . import lstm_model


rouge = evaluate.load('rouge')

def evaluate_model(
        model: lstm_model.LSTMAutoComplete, 
        tokenizer: BertTokenizerFast,
        device: str, 
        val_dataloader: DataLoader, 
        val_rouge_dataloader: DataLoader, 
        criterion: CrossEntropyLoss, 
        test_phrase: Optional[str]):
    
    model.to(device)
    model.eval()
    val_loss = 0.
#   Подсчёт validation loss
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Evaluating'):
            source = batch['source'].to(device)
            target = batch['target'].to(device)

            logits = model(source)
            loss = criterion(logits.permute(0, 2, 1), target)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)

#   Подсчёт метрик ROUGE
    inputs = []
    references = []
    generated = []
    
    with torch.no_grad():
        for batch in tqdm(val_rouge_dataloader, desc='Evaluating ROUGE'):
            max_len = batch['max_len']
            input_ids = batch['input'].to(device)
            model_outs = model.generate(input_ids, tokenizer.eos_token_id, max_len=max_len)
            
            generated.extend([
                tokenizer.decode(model_out, skip_special_tokens=True) for model_out in model_outs
                ])
            references.extend(batch['reference'])
            
    rouge_res = rouge.compute(predictions=generated, references=references)
    print(f'Validation loss: {val_loss}')
    print('Rouge metrics:')
    for k, v in rouge_res.items():
        print(f"{k}: {v:.4f}")

#   Генерация тестовой фразы для демонстрации работы модели
    if test_phrase is not None:
        with torch.no_grad():
            inputs = torch.tensor(tokenizer.encode(test_phrase, add_special_tokens=False), dtype=torch.long).to(device)
            print(test_phrase + ' ' + tokenizer.decode(
                model.generate(inputs.unsqueeze(0), tokenizer.eos_token_id, max_len=20)[0], 
                skip_special_tokens=True))
        
    return val_loss