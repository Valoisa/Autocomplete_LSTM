from transformers import pipeline, set_seed
from transformers import BertTokenizerFast

from torch.utils.data import DataLoader

from typing import List

from tqdm.auto import tqdm

import evaluate

import torch

from . import next_token_dataset
from . import lstm_model


# Считаем метрики ROUGE для GPT-2 и для LSTM и выводим оба результата
def evaluate_gpt_vs_lstm(
        model_lstm: lstm_model.LSTMAutoComplete, 
        tokenizer: BertTokenizerFast, 
        test_texts: List[str],
        device: str
        ):
    
    generator = pipeline('text-generation', model='distilgpt2', device=device)
    set_seed(42)

    test_rouge_dataset = next_token_dataset.EvalROUGEDataset(test_texts, tokenizer)
    rouge = evaluate.load('rouge')

    inputs = [item['input'] for item in test_rouge_dataset]
    references = [item['reference'] for item in test_rouge_dataset]
    generated_gpt = []
    generated_lstm = []
#   Выведем первые N дополнений текстов на экран
    n_texts = 5
    i = 0
    samples = {
        'input': [], 
        'gpt2': [],
        'lstm' : []
        }

    for input in tqdm(inputs, desc='Autocompleting inputs'):
        input_ids = tokenizer.encode(input, add_special_tokens=False)
        max_len = len(input_ids) * 2

#       Автодополнение с помощью GPT2
        gpt_out = generator(
            input, 
            max_new_tokens=len(input_ids), 
            num_return_sequences=1,
            pad_token_id=0
            )[0]['generated_text']
        gpt_autocomplete = gpt_out[len(input):].lower()
        generated_gpt.append(gpt_autocomplete)

#       Автодополнение с помощью LSTM-модели
        lstm_out = model_lstm.generate(
            torch.tensor(input_ids, dtype=torch.long),
            eos_token=tokenizer.eos_token_id,
            max_len=max_len
        )
        lstm_autocomplete = tokenizer.decode(lstm_out, skip_special_tokens=True)
        generated_lstm.append(lstm_autocomplete)

        if i < n_texts:
            samples['input'].append(input)
            samples['gpt2'].append(gpt_autocomplete)
            samples['lstm'].append(lstm_autocomplete)
            i += 1

    rouge_gpt_res = rouge.compute(predictions=generated_gpt, references=references)    
    print('Rouge metrics for test (GPT2):')
    for k, v in rouge_gpt_res.items():
        print(f"{k}: {v:.4f}")

    rouge_lstm_res = rouge.compute(predictions=generated_lstm, references=references)
    print('Rouge metrics for test (LSTM-based):')
    for k, v in rouge_lstm_res.items():
        print(f"{k}: {v:.4f}")

    print('=== Samples ===')
    for i in range(n_texts):
        print(f'Input: "{samples['input'][i]}"\n\tGPT2-autocomplete: "{samples['gpt2'][i]}"\n\tLSTM-autocomplete: "{samples['lstm'][i]}"')
        print('===================================\n')