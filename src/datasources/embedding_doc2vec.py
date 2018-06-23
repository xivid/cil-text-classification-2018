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
        print("preprocess")
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
        print(self.X[2])
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
        self.testX = np.zeros((num_tweet, 400), dtype=np.float32)

        with open(test_src) as f:
            logger.info("processing " + fn)
            for line in f:
                lines = f.readlines()
                num_tweet = len(lines)
                for idx, line in enumerate(lines):
                    self.testX[idx] = model.infer_vector(gensim.utils.simple_preprocess(line))

        print(self.testX)
        
    def train_we(self, pos_src, neg_src, save_path):
        from utils import ensure_dir
        ensure_dir(output_dir)
        print("train")

        train_corpus = []
        for fname in [pos_src, neg_src]:
            with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
                # For training data, add tags
                train_corpus += [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]) for i, line in enumerate(f)]

        max_epochs = 100
        alpha = 0.025
        
        print(train_corpus)
        
        model = gensim.models.doc2vec.Doc2Vec(vector_size=400, min_count=2, epochs=55)
        model.build_vocab(train_corpus)
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        model.save(save_path)
        #model = gensim.models.Doc2Vec.load(save_path)
        #docvecs = model.docvecs
        #print (docvecs[3])
        return model

if __name__ == "__main__":
    main()

