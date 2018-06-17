# Data Sources

Data sources, which are fed into machine learning models, are defined in this folder. Every data source (e.g. GloVe, word2vec) is implemented as a class inheriting `BaseDataSource`.

We decouple the implementation of data sources from models, so that we can try arbitrary combinations of data representations and training models, for example, GloVe+SVM, GloVe+LSTM, word2vec+SVM, word2vec+GBDT.

(TODO: This means we should remove all the additional .sh and .py files here and pack them into `glove.py`!)      