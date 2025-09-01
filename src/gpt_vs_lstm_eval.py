from transformers import pipeline, set_seed
from transformers import BertTokenizerFast

from torch.utils.data import DataLoader

from typing import List

from tqdm.auto import tqdm

import evaluate

import torch

from . import lstm_model


# Считаем метрики ROUGE для GPT-2 и для LSTM и выводим оба результата
def evaluate_gpt_vs_lstm(
        model_lstm: lstm_model.LSTMAutoComplete, 
        tokenizer: BertTokenizerFast, 
        test_dataloader: DataLoader,
        device: str,
        batch_size: int=256
        ):
    
    generator = pipeline('text-generation', model='distilgpt2', batch_size=batch_size, device=device, truncation=True)
    generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
    generator.tokenizer.padding_side = 'left'
    set_seed(42)

    rouge = evaluate.load('rouge')

    references = []
    generated_gpt = []
    generated_lstm = []
#   Выведем N дополнений текстов на экран
    n_texts = 5
    i = 0
    samples = {
        'input': [], 
        'gpt2': [],
        'lstm' : []
        }

    for batch in tqdm(test_dataloader, desc='Autocompleting inputs'):
        inputs = batch['input'].to(device)
        max_len = max(batch['max_len'])

#       Автодополнение с помощью GPT2
        gpt_inputs = [
            tokenizer.decode(input, skip_special_tokens=True)
            for input in inputs
            ]
        gpt_outputs = generator(
            gpt_inputs, 
            max_new_tokens=max_len // 3 + 1, 
            num_return_sequences=1,
            pad_token_id=0
            ) #[0]['generated_text']
        gpt_autocomplete = [
            gpt_output[0]['generated_text'][len(gpt_input):].lower() 
            for gpt_input, gpt_output in zip(gpt_inputs, gpt_outputs)
            ]
        generated_gpt.extend(gpt_autocomplete)

#       Автодополнение с помощью LSTM-модели
        lstm_outputs = model_lstm.generate(
            inputs,
            eos_token=tokenizer.eos_token_id,
            max_len=max_len
        )
        lstm_autocomplete = [
            tokenizer.decode(lstm_output, skip_special_tokens=True)
            for lstm_output in lstm_outputs
            ]
        generated_lstm.extend(lstm_autocomplete)
        
        references.extend(batch['reference'])

        if i < n_texts:
            samples['input'].append(gpt_inputs[0])
            samples['gpt2'].append(gpt_autocomplete[0])
            samples['lstm'].append(lstm_autocomplete[0])
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
    for i in range(len(samples['input'])):
        input_text = samples['input'][i]
        print(f'Input: "{input_text}"')
        gpt2_output = samples['gpt2'][i]
        print(f'\tGPT2-autocomplete: "{gpt2_output}"')
        lstm_output = samples['lstm'][i]
        print(f'\tLSTM-autocomplete: "{lstm_output}"')
        print('===================================\n')