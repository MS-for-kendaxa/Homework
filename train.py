import torch
from tqdm import tqdm
from predict import predict

class Trainer:
    def __init__(self,
                 train_sents,
                 train_labels,
                 label_tokenizer,
                 batch_size=32,
                 epochs=1,
                 force_cpu=False,
                 valid_sents=None,
                 valid_labels=None,
                 validators =(),
                 ):
        """
        A model-agnostic training class.
        :param train_sents: The training set (list of lists of words)
        :param train_labels: Training labels (list of labels in string form)
        :param label_tokenizer: Label tokenizer (for example the LabelTokenizer class)
        :param batch_size: Size of the training minibatch.
        :param epochs: Number of training epochs
        :param force_cpu: Trainer uses GPU by default if available. Set this to True to force cpu training even on a cuda-enabled machine.
        :param valid_sents: Validation set (list of lists of words) - Optional
        :param valid_labels: Validation labels (list of labels in string form) - Optional
        :param validators: A tuple of validators - subclasses of `Evaluator`.
        """
        GPU = torch.cuda.is_available()
        self.device = "cuda" if GPU and not force_cpu else "cpu"
        self.sents = train_sents
        self.labels = train_labels
        self.label_tokenizer = label_tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.valid_sents = valid_sents
        self.valid_labels = valid_labels
        self.validators = validators


    def _train_loop(self, model, opt, desc = ""):
        model.train()
        losses = []
        for start in tqdm(range(0,len(self.sents),self.batch_size),desc=desc):
            batch_sents, batch_labels = self.sents[start:start+self.batch_size], self.labels[start:start+self.batch_size]
            batch_tgt = self.label_tokenizer(batch_labels, pad=True, use_torch=True).to(self.device)

            outs = model(batch_sents)

            outs = outs.reshape(-1, outs.shape[-1])
            batch_tgt = batch_tgt.reshape(-1)

            loss = self.loss(outs, batch_tgt)
            loss.backward()

            losses.append(loss.item())

            opt.step()
            opt.zero_grad()
        return losses

    def _validate(self, model):
        preds = predict(self.valid_sents, model, output_probs=False).cpu()
        tgts = self.label_tokenizer(self.valid_labels, use_torch=True).cpu()
        metrics = {}
        for validator in self.validators:
            metrics[validator.name] = validator.eval(preds, tgts)
        return metrics

    def _log(self,log_file, epoch, loss, metrics):
        with open(log_file, "a") as f:
            for b, l in enumerate(loss):
                line = f"{epoch},{b},{l:.4f},"
                if b<len(loss)-1:
                    line += ","*len(metrics)
                else:
                    line += ",".join([f"{m:04f}" for m in metrics.values()])
                f.write(line+"\n")

    def fit(self, model, log_file, opt=None):
        """
        The main training function.
        :param model: A model wrapper. Must be callable on a list of strings, must implement `.to(device)` and `.parameters()` functions (like any Torch Module).
        :param log_file: The name of the output file. Loss is logged after every batch, metrics after every epoch.
        :param opt: Optional, optimizer for the model. If not provided, an Adam optimizer is initialized with default hyperparameters.
        :return: model, opt, metrics. Model - the trained model. opt - The optimizer. metrics - a dict of the final metrics of the model
        """
        name = model.name if hasattr(model, "name") else type(model).__name__
        model = model.to(self.device)
        if opt is None:
            opt = torch.optim.Adam(model.parameters(),lr=0.01)
        if log_file is not None:
            with open(log_file,"w+") as f:
                head = "epoch,batch,loss,"+",".join([v.name for v in self.validators])+"\n"
                f.write(head)

        for epoch in range(self.epochs):
            losses = self._train_loop(model=model, opt=opt, desc=f"{name}, epoch {epoch}:")
            metrics = self._validate(model)
            self._log(log_file, epoch, losses, metrics)
        return model, opt, metrics