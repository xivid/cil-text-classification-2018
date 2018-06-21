"""Model definitions (one class per file) to define NN architectures."""
from .bayes import NaiveBayes
#from .svm import SVM
from .new_svm import SVM
from .lightgbm import LightGBM

__all__ = ('NaiveBayes', 'SVM', 'LightGBM')


def get_model(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "SVM":
        return SVM
    elif name == 'LightGBM':
        return LightGBM
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)
