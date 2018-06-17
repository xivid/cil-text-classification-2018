class BaseDataSource:
    def preprocess(self):
        """
        preprocess the data source file
        """
        raise NotImplementedError()

    def yield_one_sample(self):
        """
        yield (the feature vector and the corresponding class label) of one sample on each call to this function.
        """
        raise NotImplementedError()
    
    def create_embeddings(self):
        """
        create word embeddings
        """
        raise NotImplementedError()

    def transform(self):
        """
        generate feature vectors based on word embeddings 
        """
        raise NotImplementedError()
    """
    TODO: add any other methods that may be useful, 
          e.g. yield_multiple_samples(n) (new requirements may come up as we proceed)
    """