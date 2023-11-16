import torch
import torch.nn as nn
import torch.nn.functional as F


class BertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        self.linear = nn.Linear(768, 3)

    def forward(self, tokens_tensor, segments_tensors, attention_tensor):
        sent_embeds = self.bert(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)

        encoded_layers = sent_embeds[0]

        out = encoded_layers.mean(dim=1)
        out = self.linear(out)
        # out =      # Cross-Entropy Loss takes in the unnormalized logits

        return out




class LSTM_entailer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        pretrained_fasttext_embeddings  = torch.rand((vocab_size,embedding_dim))
        self.word_embeddings.weight.data.copy_(pretrained_fasttext_embeddings)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)


    def forward(self, sentence1, sentence2):
        embeds1 = self.word_embeddings(sentence1)
        embeds2 = self.word_embeddings(sentence2)

        out1, _ = self.lstm(embeds1.view(len(sentence1), 1, -1))
        out2, _ = self.lstm(embeds2.view(len(sentence2), 1, -1))

        out1 = out1[-1]
        out2 = out2[-1]

        return out1, out2