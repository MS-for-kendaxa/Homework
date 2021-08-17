import numpy as np
import torch


class Evaluator():
    """
        base evaluator. Subclasses must implement the `_eval_fn` function, and have property `name`
    """
    def _eval_fn(self, preds: torch.Tensor, target_labels):
        """
        Evaluates cleaned predictions. Subclasses must implement this.
        :param preds: a 1-dimensional torch tensor of class predictions. Does not include any padding.
        :param target_labels: A 1-dimensional torch tensor of true labels. Does not include any padding.
        :return:
        """
        raise NotImplemented

    def eval(self, preds, target_labels):
        preds = preds.reshape(-1)
        target_labels = target_labels.reshape(-1)
        mask = (target_labels==0)
        preds = preds[~mask]
        target_labels = target_labels[~mask]
        return self._eval_fn(preds, target_labels)

class EntityAccuracy(Evaluator):
    """
        Exact one-to-one accuracy. A difference in entity or a difference in entity start/end counts as a miss.
        IMPORTANT: Does not count any correctly predicted non-entites
    """
    def __init__(self):
        self.name = "Exact Accuracy"

    def _eval_fn(self, preds, target_labels):
        mask = (preds==target_labels)&(preds==1)
        preds, target_labels = preds[~mask], target_labels[~mask]
        return (preds==target_labels).sum().item()/len(target_labels)

class BinaryAccuracy(Evaluator):
    """
    Accuracy in entity prediction. Disregards the type of entity.
    """
    def __init__(self):
        self.name = "Binary Accuracy"

    def _eval_fn(self, preds, target_labels):
        preds_entities = preds != 1
        target_entities = target_labels!=1
        return (preds_entities==target_entities).sum().item()/len(target_labels)

class BinaryF1(Evaluator):
    """
    F1 score of entity prediction. Disregards the type of entity.
    """
    def __init__(self):
        self.name = "Binary F1"

    def _eval_fn(self, preds, target_labels):
        preds_entities = preds != 1
        target_entities = target_labels != 1
        if preds_entities.sum()==0:
            return 0
        precision = (preds_entities & target_entities).sum().item()/preds_entities.sum().item()
        recall = (preds_entities & target_entities).sum().item()/target_entities.sum().item()
        return 2*(precision*recall)/(precision+recall)