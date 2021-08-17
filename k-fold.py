import os
from models import LSTMWrapper, FasttextEmbedder
from train import Trainer
from tokenizers import LabelTokenizer
from eval import *
from data_loading import load_data
import argparse
import random
from copy import deepcopy, copy
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=10, type=int)
args = parser.parse_args()

def k_fold(model,data,labels, k, shuffle_data=False, validators=(), *args, **kwargs):
    """
    Makes k copies of the model and cross-validates them using a Trainer class.
    args and kwargs are passed to the Trainer class.
    Original model is NOT updated.

    :param model:
    :param data:
    :param labels:
    :param k:
    :param shuffle_data:
    :param validators:
    :return: k trained copies of the original model, k dicts with metric
    """
    models = []
    label_tokenizer = LabelTokenizer(labels)
    if shuffle_data:
        dl = list(zip(data,labels))
        random.shuffle(dl)
        data, labels = list(*zip(dl))
    for i in range(k):
        model_i = model.get_deepcopy()
        model_i.name+="_fold_%s"%i
        models.append(model_i)
    results = []
    for i in range(k):
        valid_start = len(data)*i//k
        valid_end = valid_start+len(data)//k
        valid_data, valid_labels = data[valid_start:valid_end], labels[valid_start:valid_end]
        train_data = data[:valid_start] + data[valid_end:]
        train_labels = labels[:valid_start] + labels[valid_end:]
        trainer = Trainer(
            train_data,
            train_labels,
            label_tokenizer,
            valid_sents=valid_data,
            valid_labels=valid_labels,
            validators=validators,
            *args,
            **kwargs
        )
        os.makedirs(f"{k}_fold", exist_ok=True)
        _, _, mets = trainer.fit(models[i], f"{k}_fold/model_{i}.log")
        models[i] = models[i].cpu()
        torch.cuda.empty_cache()
        print(mets)
        results.append(mets)
    return models, mets

if __name__=="__main__":
    data, labels = load_data("dataset.txt")
    # fasttext_embed = FasttextEmbedder("pretrained_100.bin")

    model = LSTMWrapper(data,LabelTokenizer(labels).num_labels)
    # model = LSTMWrapper(data, LabelTokenizer(labels).num_labels, pretrained_embeddings=fasttext_embed,
    #                              hidden_size=128, name="Fasttext model")

    k_fold(model, data, labels, k=args.folds, validators=(EntityAccuracy(), BinaryAccuracy(), BinaryF1()))
