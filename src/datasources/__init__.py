from .glove import GloVe
from .embedding_word2vec import Word2VecEmbedding
from .embedding_doc2vec import Doc2VecEmbedding
__all__ = ("GloVe", "Word2VecEmbedding", "Doc2VecEmbedding")


def get_datasource(name):
    if name == "GloVe":
        return GloVe
    elif name == "Word2VecEmbedding":
        return Word2VecEmbedding
    elif name == "Doc2VecEmbedding":
        return Doc2VecEmbedding
    else:
        raise NotImplementedError("No data source called \"%s\"!" % name)
