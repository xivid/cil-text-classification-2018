"""Model definitions (one class per file) to define NN architectures."""
from .bayes import NaiveBayes
#from .svm import SVM
from .new_svm import SVM
from .lightgbm import LightGBM
#from .lstm import LSTM

#__all__ = ('NaiveBayes', 'SVM', 'LightGBM', 'LSTM')
__all__ = ('NaiveBayes', 'SVM', 'LightGBM', 'LSTM')


def get_model(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "SVM":
        return SVM
    elif name == 'LightGBM':
        return LightGBM
    # elif name == 'LSTM':
    #     return LSTM
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)
