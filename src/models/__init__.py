"""Model definitions (one class per file) to define NN architectures."""
#from .bayes import NaiveBayes
from .svm import Svm

#__all__ = ('NaiveBayes', )
__all__ = ('Svm', )

def getModel(name):
    if name == "NaiveBayes":
        return NaiveBayes
    elif name == "Svm":
        return Svm
    else:
        raise NotImplementedError("No model called \"%s\"!" % name)
