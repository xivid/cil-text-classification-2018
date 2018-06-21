class BaseDataSource(object):

    def __init__(self, **kwargs):
        self.X = None
        self.Y = None
        self.testX = None
        self.preprocess(**kwargs)

    def preprocess(self, **kwargs):
        """
        preprocess the data source file, store feature vectors in self.X, self.testX and labels in self.Y
        """
        raise NotImplementedError()
