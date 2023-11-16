import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import clean_text, le

class Dataset(Dataset):
    def __init__(self, df):
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

        tokens = []
        segments = []
        masks = []
        labels = []

        for idx in df.index:
            sentence1, sentence2, y = df.loc[idx, ['sentence1', 'sentence2', 'gold_label']]
            sentence1, sentence2, y = clean_text(sentence1), clean_text(sentence2), le(y)

            indexed_tokens = tokenizer.encode(sentence1, sentence2, add_special_tokens=True, padding='max_length')
            segments_ids = np.ones_like(indexed_tokens)
            attention_mask = np.zeros_like(indexed_tokens)

            segments_ids[:indexed_tokens.index(102)+1] = 0
            attention_mask[:indexed_tokens.index(0)] = 1
            tokens.append(np.array(indexed_tokens))
            segments.append(segments_ids)
            masks.append(attention_mask)
            labels.append(y)

        self.tokens = np.array(tokens)[:, :50]
        self.segments = np.array(segments)[:, :50]
        self.masks = np.array(masks)[:, :50]
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        indexed_tokens = torch.tensor(self.tokens[idx])
        segments_id = torch.tensor(self.segments[idx])
        attention_mask = torch.tensor(self.masks[idx])
        y = torch.tensor(self.labels[idx])

        return indexed_tokens, segments_id, attention_mask, y


        # sentence1, sentence2, y = self.df.loc[idx, ['sentence1', 'sentence2', 'gold_label']]
        # sentence1, sentence2, y = clean_text(sentence1), clean_text(sentence2), le(y)

        # indexed_tokens = self.tokenizer.encode(sentence1, sentence2, add_special_tokens=True, padding='max_length')
        # segments_ids = np.ones_like(indexed_tokens)
        # attention_mask = np.zeros_like(indexed_tokens)

        # segments_ids[:indexed_tokens.index(102)+1] = 0
        # attention_mask[:indexed_tokens.index(0)] = 1

        # segments_tensors = torch.tensor(np.array(segments_ids))
        # attention_tensor = torch.tensor(np.array(attention_mask))
        # tokens_tensor = torch.tensor([indexed_tokens])[0]

        return tokens_tensor, segments_tensors, attention_tensor, y