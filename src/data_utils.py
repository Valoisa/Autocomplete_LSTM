import re

import unicodedata


mention = re.compile(r'@\w+ ')
url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\\\/+~#=]+')
emoticon = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P|O|X|p|x|o)+')
kaomoji = re.compile(r'(?:x|X|o|O|t|T)(?:_|-|\.)?(?:x|X|o|O|t|T)')
spacios_space = re.compile(r'\s+')

def remove_urls(text: str):
    return url.sub('', text)

def remove_mentions(text: str):
    return mention.sub('', text)

def remove_emojis(text: str):
    return emoticon.sub('', kaomoji.sub('', text))

def remove_long_spaces(text: str):
    return spacios_space.sub(' ', text).strip()

def remove_diacritics(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def clean_text(text: str):
    text = remove_mentions(text)
    # removing urls and text emojis to prevent stanalone letters and url parts
    text = remove_urls(text)
    text = remove_emojis(text) 
    text = remove_long_spaces(text)
    text = remove_diacritics(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text