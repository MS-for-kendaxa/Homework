import torch
# from transformers import PreTrainedTokenizer
from collections import defaultdict
from collections import Counter

class DefaultTokenizer:
    """
    Turns strings into integers
    """
    def __init__(self, data):
        word_vocab = [d[0] for d in Counter([w for sent in data for w in sent]).most_common()]
        self.special_tokens = ["<PAD>", "<OOV>"]
        self.word_vocab = self.special_tokens + word_vocab
        self.reversed_vocab = {}#defaultdict(lambda: 1)
        for v in self.word_vocab:
            self.reversed_vocab[v] = len(self.reversed_vocab)

    @property
    def output_size(self):
        return len(self.word_vocab)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.word_vocab[item]
        else:
            return self.reversed_vocab[item]

    def tokenize_sents(self, sents):
        """
        Tokenizes the given sentences
        :param sents: list of lists of strings to be tokenized
        :return: (Tokens, lengths) Tokens is a (padded) pytorch tensor of integers, shape max_length x batch_size.
        Lengths is a list containing lengths of sentences.
        """
        tokens = [[self.reversed_vocab[w] for w in sent] for sent in sents]
        lens = [len(t) for t in tokens]
        max_len = max(lens)
        tokens = [[t[i] if i<len(t) else 0 for i in range(max_len)] for t in tokens]

        return torch.Tensor(tokens).long().T, lens

    def __call__(self, *args, **kwargs):
        return self.tokenize_sents(*args, **kwargs)

class LabelTokenizer:
    """
    Converts labels to integers and back.
    """
    def __init__(self, labels):
        label_vocab = [d[0] for d in Counter([l for labs in labels for l in labs]).most_common()]
        self.label_vocab = ["<PAD>"]+label_vocab
        self.reversed_label_vocab = defaultdict(int)
        for l in self.label_vocab:
            self.reversed_label_vocab[l] = len(self.reversed_label_vocab)

    @property
    def num_labels(self):
        return len(self.label_vocab)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.label_vocab[item]
        else:
            return self.label_vocab[item]

    def tokenize_labels(self, labels, pad=True, use_torch=False):
        labels = [[self.reversed_label_vocab[l] for l in labs] for labs in labels]
        max_len = max([len(l) for l in labels])
        if pad:
            labels = [[l[i] if i<len(l) else 0 for i in range(max_len)] for l in labels]
        if use_torch:
            return torch.Tensor(labels).long().T
        else:
            return labels


    def detokenize_labels(self, tokens, include_pad=False):
        labels = []
        for l in tokens:
            labels.append([self.label_vocab[tok] for tok in l if (tok != 0) or include_pad])
        return labels

    def __call__(self, *args, **kwargs):
        return self.tokenize_labels(*args, **kwargs)