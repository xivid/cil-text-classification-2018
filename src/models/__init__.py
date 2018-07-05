"""Model definitions (one class per file) to define NN architectures."""
from .bayes import NaiveBayes
from .logistic import LR
from .svm import SVM
from .lightgbm import LightGBM
#from .lstm import LSTMModel

__all__ = ('NaiveBayes', 'LR', 'SVM', 'LightGBM')


def get_model(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "LR":
        return LR
    elif name == "SVM":
        return SVM
    elif name == 'LightGBM':
        return LightGBM
    #elif name == 'LSTM':
    #    return LSTMModel
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)
