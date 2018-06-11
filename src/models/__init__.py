"""Model definitions (one class per file) to define NN architectures."""
from .bayes import NaiveBayes

__all__ = ('NaiveBayes', )

def getModel(name):
    if name == "NaiveBayes":
        return NaiveBayes
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)