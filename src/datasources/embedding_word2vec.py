from core import BaseDataSource
from gensim.models import Word2Vec, KeyedVectors
from utils.feature_extraction import average_vector
from pathlib import Path
import logging

pos_filename = '../../data/train_pos.txt'
neg_filename = '../../data/train_neg.txt'
test_filename = '../../data/test_data_stripped.txt'

logger = logging.getLogger("Word2Vec")
output_dir = "../output/datasources/Word2Vec/"

class Word2VecEmbedding(BaseDataSource):
    """
    This Word2VecEmbedding class train the word embeddings in preprocess()->train_we()
    (and output the intermediate results to ../../output/datasources/GloVe/),
    and yield (the feature vector and the corresponding class label of) one sample on
    each call to yield_one_sample(), or expose all samples with self.X, self.Y.
    """

    def preprocess(self, pos_src, neg_src, test_src, embedding_src=""):
        if embedding_src != "":
            # load pre-trained embeddings
            logger.info("Loading external embeddings")
            word_vectors = KeyedVectors.load_word2vec_format(embedding_src)
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
        sentences = []
        maxLen = 0;
        for fn in [pos_src, neg_src]:
            with open(fn) as f:
                for line in f:
                    splitted = line.split()
                    if maxLen < len(splitted):
                        maxLen = len(splitted)
                    sentences.append(splitted)

        logger.info("Starting word2vec")
        feature_dimensions = 20
        model = Word2Vec(sentences, size=feature_dimensions)
        model.wv.save_word2vec_format(save_path)

        return model
