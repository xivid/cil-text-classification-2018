from core import BaseDataSource
from pathlib import Path
import logging
import os
import gensim
import os
import collections
import smart_open
import random
import numpy as np

logger = logging.getLogger("Doc2Vec")
output_dir = "../output/datasources/Doc2Vec/"

class Doc2VecEmbedding(BaseDataSource):
    """
    This Doc2VecEmbedding class train the document embeddings in preprocess()->train_we()
    (and output the intermediate results to ../../output/datasources/Doc2Vec/),
    and yield (the feature vector and the corresponding class label of) one sample on
    each call to yield_one_sample(), or expose all samples with self.X, self.Y.
    """
    def preprocess(self, pos_src, neg_src, test_src, embedding_src=""):
        logger.info("preprocessing")
        if embedding_src != "":
            # load pre-trained embeddings
            logger.info("Loading external embeddings")
            file_type = os.path.splitext(embedding_src)[1]
            if file_type == ".model":
                model = gensim.models.Doc2Vec.load(embedding_src)
            else:
                logger.error("Unsupported file type!")


        else:
            saved_path = output_dir + "doc2vec_word_embedding.model"
            saved = Path(saved_path)
            if saved.is_file():
                logger.info("Loading cached embeddings from " + saved_path)
                model = gensim.models.Doc2Vec.load(saved_path)

            else:
                logger.info("Training doc embeddings and saving to " + saved_path)
                model = self.train_we(pos_src, neg_src, saved_path)

        self.X = model.docvecs
        logger.info("self.X: len %d" % len(self.X))
        Y = np.array([])

        for fn in [pos_src, neg_src]:
            with open(fn) as f:
                logger.info("processing " + fn)
                lines = f.readlines()
                num_tweet = len(lines)

                if Y.size == 0:
                    Y = np.array([1] * num_tweet)
                else:
                    Y = np.concatenate((Y, [-1] * num_tweet))
        self.Y = Y

        with open(test_src) as f:
            logger.info("processing " + test_src)
            lines = f.readlines()
            num_tweet = len(lines)
            self.testX = np.zeros((num_tweet, 300), dtype=np.float32)

            for idx, line in enumerate(lines):
                    self.testX[idx] = model.infer_vector(gensim.utils.simple_preprocess(line))

        # print(self.testX)
        
    def train_we(self, pos_src, neg_src, save_path):
        from utils import ensure_dir
        ensure_dir(output_dir)
        logger.info("building corpus")

        train_corpus = []
        i = 0  # tag number
        for fname in [pos_src, neg_src]:
            with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
                # For training data, add tags
                logger.info("processing " + fname)
                for line in f:
                    train_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]))
                    i += 1
                    if i % 10000 == 0:
                        logger.info("added %d lines" % i)

        logger.info("built train_corpus of length %d" % len(train_corpus))
        # print(train_corpus)

        epochs = 20  # typically 10~20
        model = gensim.models.doc2vec.Doc2Vec(train_corpus, vector_size=300, min_count=2, epochs=epochs)
        model.save(save_path)

        return model


