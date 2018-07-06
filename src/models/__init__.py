"""Model definitions (one class per file) to define models."""
from .naive_bayes import NaiveBayes
from .lr import LR
from .svm import SVM
from .lightgbm import LightGBM

__all__ = ('NaiveBayes', 'LR', 'LightGBM')


def get_model(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "LR":
        return LR
    elif name == 'LightGBM':
        return LightGBM
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)
