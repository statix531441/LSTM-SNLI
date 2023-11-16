import re
import string
import torch

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def le(gold):
    if gold=='neutral': return 1
    elif gold=='entailment': return 2
    else: return 0

def prepare_sequence(sentence, vocab):
    idxs = [vocab[word] if word in vocab else 0 for word in sentence]
    return torch.tensor(idxs, dtype=torch.long)