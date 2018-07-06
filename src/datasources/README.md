# Data Sources

Data sources, which are fed into machine learning models, are defined in this folder. Every data source (e.g. GloVe, word2vec) is implemented as a class inheriting `BaseDataSource`.

We decouple the implementation of data sources from models, so that we can try different combinations of data representations and training models, for example, GloVe+LR, word2vec+GBDT, WordOccurrence+NaiveBayes.

For neural network models, since the data pre-processing and word embeddings are sometimes specific to different models, we provide a `RawText` data source to feed raw text data into the model, and let the model handle the embedding. 

