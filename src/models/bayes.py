from core import BaseModel
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB


class NaiveBayes(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.vocabulary = dict()
        """ X
            Word occurrence vectors, each row is a tweet, each column is a word.
            X[i][j] is the times by which the word j occurs in the tweet i.
            Use sparse matrix to significantly reduce memory usage.
        """
        self.X = None
        """ Y
            Class labels. 1 for positive, -1 for negative.
        """
        self.Y = None
        """ model
            Trained model.
        """
        self.model = None

    def load_training_data(self, pos_src, neg_src):
        print("loading positive samples...")

        # read lines
        f = open(pos_src, "r")
        lines = f.readlines()
        n_lines = len(lines)
        print("%s: %d samples" % (pos_src, n_lines))

        # class labels
        self.Y = [1] * n_lines

        # gather information for sparse matrix
        indptr = [0]
        indices = []
        data = []
        cnt = 0
        for line in lines:
            tokens = line.strip().split()
            for token in tokens:
                index = self.vocabulary.setdefault(token, len(self.vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            cnt += 1
            if cnt % 100 == 0:
                print("%d/%d loaded" % (cnt, n_lines))

        f.close()
        del lines

        print("loading negative samples...")

        # read lines
        f = open(neg_src, "r")
        lines = f.readlines()
        n_lines = len(lines)
        print("%s: %d samples" % (pos_src, n_lines))

        # class labels
        self.Y += [-1] * n_lines

        # append sparse matrix
        cnt = 0
        for line in lines:
            tokens = line.strip().split()
            for token in tokens:
                index = self.vocabulary.setdefault(token, len(self.vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            cnt += 1
            if cnt % 100 == 0:
                print("%d/%d loaded" % (cnt, n_lines))
        print("%d loaded" % n_lines)

        # build sparse matrix
        self.X = csr_matrix((data, indices, indptr), dtype=int)

        self.Y = np.array(self.Y)

    def train(self, pos_src, neg_src):
        self.load_training_data(pos_src, neg_src)
        clf = BernoulliNB()
        clf.fit(self.X, self.Y)
        self.model = clf
        print("Trained model: " + str(self.model))

    def predict(self, test_src):
        print("loading test data...")

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
                index = self.vocabulary.get(token, -1)
                if index != -1:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))
            cnt += 1
            if cnt % 100 == 0:
                print("%d/%d loaded" % (cnt, n_lines))

        # build sparse matrix
        mat = csr_matrix((data, indices, indptr), shape=(n_lines, len(self.vocabulary)), dtype=int)

        return self.model.predict(mat)
