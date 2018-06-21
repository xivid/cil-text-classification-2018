from .glove import GloVe
from .embedding_word2vec import Word2VecEmbedding
__all__ = ("GloVe", "Word2VecEmbedding")


def get_datasource(name):
    if name == "GloVe":
        return GloVe
    elif name == "Word2VecEmbedding":
        return Word2VecEmbedding
    else:
        raise NotImplementedError("No data source called \"%s\"!" % name)
