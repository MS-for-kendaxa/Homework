import copy

import torch
import torch.nn.functional as F
from tokenizers import DefaultTokenizer
from collections import Counter
import fasttext
import numpy as np


class LSTMClassifier(torch.nn.Module):
    """
    A basic LSTM predictor. Uses bidirectional LSTM and a single feedforward layer. Can use pretrained embeddings, or can train its own.
    """
    def __init__(self, num_classes, input_size, hidden_size=512, pretrained_embeddings=True, embedding_size=256):
        super().__init__()
        lstm_inp_size = embedding_size if not pretrained_embeddings else input_size
        if not pretrained_embeddings:
            self.embedding = torch.nn.Embedding(input_size, embedding_dim=embedding_size,padding_idx=0)
        else:
            self.embedding = None
        self.lstm = torch.nn.LSTM(hidden_size=hidden_size, input_size=lstm_inp_size, bidirectional=True, dropout=0.5, num_layers=2)
        self.classifier = torch.nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, lens):
        if self.embedding is not None:
            x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lens,enforce_sorted=False)
        x = self.lstm(x)[0]
        x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
        x = self.classifier(x)
        #x = F.softmax(x, dim=-1)
        return x

class LSTMWrapper(torch.nn.Module):
    def __init__(self, data, num_labels, pretrained_embeddings=None, name="Basic LSTM", *args, **kwargs):
        super().__init__()
        self.name = name
        self.num_labels = num_labels
        self.pretrained_embeddings = pretrained_embeddings
        self.lstm_args = args
        self.lstm_kwargs = kwargs
        if pretrained_embeddings is not None:
            self.preprocess = pretrained_embeddings
        else:
            self.preprocess = DefaultTokenizer(data)
        self.lstm_model = LSTMClassifier(num_labels,
                                         self.preprocess.output_size,
                                         *args,
                                         pretrained_embeddings=pretrained_embeddings is not None,
                                         **kwargs)

    def __call__(self, sents):
        model_inp, lens = self.preprocess(sents)
        model_inp = model_inp.to(self.parameters().__next__().device)
        return self.lstm_model(model_inp, lens)

    def get_deepcopy(self):
        cp = LSTMWrapper([],1, name=self.name, hidden_size = 1, embedding_size = 1)
        cp.lstm_model = copy.deepcopy(self.lstm_model)
        cp.preprocess = self.preprocess
        return cp

    def save_model(self, path):
        torch.save(self, path)

class FasttextEmbedder:
    def __init__(self, file_loc):
        self.embed = fasttext.FastText.load_model(file_loc)

    @property
    def output_size(self):
        return self.embed.get_dimension()

    def __call__(self, text):
        vecs = [[self.embed.get_word_vector(w) for w in sent] for sent in text]
        lens = [len(v) for v in vecs]
        max_len = max(lens)
        vecs = [ v + [np.full_like(v[0],0) for _ in range(max_len-len(v))] for v in vecs]
        vecs_tensor = torch.Tensor(vecs)
        return vecs_tensor.permute(1,0,2), lens