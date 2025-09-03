# Autocomplete_LSTM
Учебный проект для курса Deep Learning, Спринт 2.

Данный репозиторий содержит код для обучения и оценки модели машинного обучения на базе LSTM. 
Это модель принимает на вход токенизированный текст и возвращает "продолжение" (автодополнение).
Архитектура модели:
* Embedding слой: input_size=vocab_size, output_size=hidden_size
* LSTM слой: hidden_size=512, num_layers=2
* Dropout слой: p=0.3
* Feed-forward слой: input_size=hidden_size, output_size=vocab_size

Токенизация посимвольная.

Файлы проекта:
* scr
  * char_tokenizer.py: определение класса CharTokenizer
  * data_utils.py: методы для предобработки текста
  * next_token_dataset.py: определение класса датасетов для обучения и оценкм
  * lstm_model.py: определение модели
  * model_train.py: метод для обучения модели (в течение одной эпохи)
  * model_eval.py: метод для оценки работы модели
  * gpt_vs_lstm_eval.py: методы для сравнения перфоманса gpt2 и модели на базе lstm
* data
  * raw_dataset.csv: все тексты из скачанного датасета до обработки
  * processed_dataset.csv: тексты после обработки
  * train.csv, val.csv, test.csv: разбиение датаеста на трейн/валидациюю/тест 
* checkpoints
  * best_model.pth: веса модели, сохранённые во время/после обучения.   
