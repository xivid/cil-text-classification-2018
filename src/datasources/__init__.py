from .glove import GloVe

__all__ = ("GloVe", )


def get_datasource(name):
    if name == "GloVe":
        return GloVe

    else:
        raise NotImplementedError("No data source called \"%s\"!" % name)
