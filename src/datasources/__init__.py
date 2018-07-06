from .glove import GloVe
from .word2vec import Word2Vec
from .doc2vec import Doc2Vec
from .raw_text import RawText
__all__ = ("GloVe", "Word2Vec", "Doc2Vec", "RawText")


def get_datasource(name):
    if name == "GloVe":
        return GloVe
    elif name == "Word2Vec":
        return Word2Vec
    elif name == "Doc2Vec":
        return Doc2Vec
    elif name == "RawText":
        return RawText
    else:
        raise NotImplementedError("No data source called \"%s\"!" % name)
