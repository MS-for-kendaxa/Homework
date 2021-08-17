import os
from models import LSTMWrapper, FasttextEmbedder
from train import Trainer
from tokenizers import LabelTokenizer
from eval import *
from data_loading import load_data
import argparse
from collections import defaultdict
import math

parser = argparse.ArgumentParser()
parser.add_argument("--grid_search", default=False)
args = parser.parse_args()

if __name__ == '__main__':
    data_clean, labels = load_data("dataset.txt")

    label_tokenizer = LabelTokenizer(labels)
    # fasttext_embed = FasttextEmbedder("pretrained_100.bin")
    split = math.ceil(len(labels)*0.9)
    trainer = Trainer(
        data_clean[:split],
        labels[:split],
        label_tokenizer,
        epochs=2,
        valid_sents=data_clean[split:],
        valid_labels=labels[split:],
        validators=(EntityAccuracy(), BinaryAccuracy(), BinaryF1())
    )
    if not args.grid_search:
        model_base = LSTMWrapper(data_clean, label_tokenizer.num_labels, hidden_size=128, embedding_size=128)
        # model_fasttext = LSTMWrapper(data_clean, label_tokenizer.num_labels, pretrained_embeddings=fasttext_embed, hidden_size=128, name="Fasttext model")
        model_base, _, mets = trainer.fit(model_base, log_file="base_lstm.log")
        print(mets)
        # _, _, mets = trainer.fit(model_fasttext, log_file="fasttext_lstm.log")
        # print(mets)
        model_base.save_model("model.pickle")
    else:
        best_metrics = defaultdict(lambda: ("",0))
        for epoch in range(1,5):
            trainer.epochs = epoch
            for batch in [2**x for x in range(3,6)]:
                trainer.batch_size = batch
                for hidden_size in [2**x for x in range(5, 9)]:
                    for embed_size in [2**x for x in range(5,9)]:
                        name = f"LSTM_hs-{hidden_size}_es-{embed_size}_batch-{batch}_epochs-{epoch}"
                        model = LSTMWrapper(data_clean,
                                            label_tokenizer.num_labels,
                                            hidden_size=hidden_size,
                                            embedding_size=embed_size,
                                            name=name)
                        os.makedirs("grid_search", exist_ok=True)
                        _, _, mets = trainer.fit(model, f"grid_search/{name}.log")
                        print(mets)
                        for k, m in mets.items():
                            if best_metrics[k][1]<m:
                                best_metrics[k]=(name,m)
        print(best_metrics)
