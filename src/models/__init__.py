"""Model definitions (one class per file) to define models."""
from .naive_bayes import NaiveBayes
from .lr import LR
from .svm import SVM
from .lightgbm import LightGBM
from .elmo_multi_rnn import ELMoMultiRNN
from .biLSTM import BiLSTM
from .elmo_GRU import ELMoGRU
from .multi_biLSTM import MultiBiLSTM
from .elmo_biLSTM import ELMoBiLSTM

__all__ = ('NaiveBayes', 'LR', 'SVM', 'LightGBM', 'ELMoMultiRNN', 'BiLSTM', 'ELMoGRU', 'MultiBiLSTM', 'ELMoBiLSTM')


def get_model(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "LR":
        return LR
    elif name == 'SVM':
        return SVM
    elif name == 'LightGBM':
        return LightGBM
    elif name == 'ELMoMultiRNN':
        return ELMoMultiRNN
    elif name == 'BiLSTM':
        return BiLSTM
    elif name == 'ELMoGRU':
        return ELMoGRU
    elif name == 'MultiBiLSTM':
        return MultiBiLSTM
    elif name == 'ELMoBiLSTM':
        return ELMoBiLSTM
    else:
        raise NotImplementedError("No model named \"%s\"!" % name)
