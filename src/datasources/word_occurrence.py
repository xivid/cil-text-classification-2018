from core import BaseDataSource
import logging
import numpy as np
from scipy.sparse import csr_matrix
from utils.feature_extraction import load_sparse_tuples_from_file

logger = logging.getLogger("WordOccurrence")


class WordOccurrence(BaseDataSource):
    """
    Word occurrence matrix
    Each row represents a tweet, each column represents a word in the whole corpus.
    X[i][j] is the times by which the word j occurs in the tweet i.
    Uses sparse matrix to significantly reduce memory usage.
    """
    def process(self, pos_src, neg_src, test_src):
        logger.info("Loading training data")
        vocabulary = self.load_training_data(pos_src, neg_src)
        logger.info("Loading testing data")
        self.load_testing_data(test_src, vocabulary)

    def load_training_data(self, pos_src, neg_src):
        # gather information for sparse matrix
        vocabulary = dict()
        indptr = [0]
        indices = []
        data = []

        # load the data
        logger.info("Loading positive samples...")
        n_lines = load_sparse_tuples_from_file(pos_src, vocabulary, data, indices, indptr)
        self.Y = [1] * n_lines
        logger.info("Loading negative samples...")
        n_lines = load_sparse_tuples_from_file(neg_src, vocabulary, data, indices, indptr)
        self.Y += [-1] * n_lines

        # build sparse matrix
        self.X = csr_matrix((data, indices, indptr), dtype=int)
        self.Y = np.array(self.Y)
        return vocabulary

    def load_testing_data(self, test_src, vocabulary):
        # read lines
        f = open(test_src, "r")
        lines = f.readlines()
        n_lines = len(lines)
        print("%s: %d samples" % (test_src, n_lines))

        # gather information for sparse matrix
        indptr = [0]
        indices = []
        data = []
        cnt = 0
        for line in lines:
            line = line.strip()
            noptr = line.find(",")
            line = line[noptr + 1:]
            tokens = line.strip().split()
            for token in tokens:
                index = vocabulary.get(token, -1)
                if index != -1:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))
            cnt += 1
            if cnt % 1000 == 0:
                print("%d/%d loaded" % (cnt, n_lines))

        # build sparse matrix
        self.testX = csr_matrix((data, indices, indptr), shape=(n_lines, len(vocabulary)), dtype=int)
