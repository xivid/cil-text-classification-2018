from core import BaseDataSource
from gensim.models import Word2Vec as Word2VecModel, KeyedVectors
from utils.feature_extraction import average_vector
from pathlib import Path
import logging
import os

logger = logging.getLogger("Word2Vec")
output_dir = "../output/datasources/Word2Vec/"

class Word2Vec(BaseDataSource):
    """
    This Word2Vec class train the word embeddings in preprocess()->train_we()
    (and output the intermediate results to ../../output/datasources/GloVe/),
    and expose (the feature vector and the corresponding class label of) all samples with self.X, self.Y.
    """

    def process(self, pos_src, neg_src, test_src, embedding_src=""):
        if embedding_src != "":
            # load pre-trained embeddings
            logger.info("Loading external embeddings")
            file_type = os.path.splitext(embedding_src)[1]
            if file_type == ".bin":
                is_binary = True
            elif file_type == ".txt":
                is_binary = False
            else:
                logger.error("Unsupported file type!")
            word_vectors = KeyedVectors.load_word2vec_format(embedding_src, binary = is_binary)
                
        else:
            saved_path = output_dir + "word2vec_word_embedding.txt"
            saved = Path(saved_path)
            if saved.is_file():
                logger.info("Loading cached embeddings from " + saved_path)
                word_vectors = KeyedVectors.load_word2vec_format(saved_path)
            else:
                logger.info("Training word embeddings and saving to " + saved_path)
                word_vectors = self.train_we(pos_src, neg_src, saved_path)
        logger.info("Calculating average vectors")
        self.X, self.Y, self.testX = average_vector(
            pos_src=pos_src,
            neg_src=neg_src,
            test_src=test_src,
            embedding=word_vectors)

    def train_we(self, pos_src, neg_src, save_path):
        from utils.io import ensure_dir
        ensure_dir(save_path)

        sentences = []
        maxLen = 0
        for fn in [pos_src, neg_src]:
            with open(fn) as f:
                for line in f:
                    splitted = line.split()
                    if maxLen < len(splitted):
                        maxLen = len(splitted)
                    sentences.append(splitted)

        logger.info("Training word2vec")
        feature_dimensions = 300
        model = Word2VecModel(sentences, size=feature_dimensions, iter=20)
        model.wv.save_word2vec_format(save_path)

        return model
