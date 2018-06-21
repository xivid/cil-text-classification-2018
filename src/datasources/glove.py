#!/usr/bin/env python3
from core import BaseDataSource
from scipy.sparse import *
import numpy as np
from utils.feature_extraction import average_vector
from gensim.models import KeyedVectors
from pathlib import Path


class GloVe(BaseDataSource):
    """
    This GloVe class train the word embeddings in preprocess()->train_we() (and output the intermediate results to ../../output/datasources/GloVe/),
    and yield (the feature vector and the corresponding class label of) one sample on each call to yield_one_sample(),
    or expose all samples with self.X, self.Y.
    """

    def preprocess(self, pos_src, neg_src, test_src, embedding_src=""):
        if embedding_src != "":
            # load pre-trained embeddings
            word_vectors = KeyedVectors.load_word2vec_format(embedding_src)
        else:
            saved_path = "../output/datasources/" + self.__class__.__name__ + "/embeddings.npz"
            saved = Path(saved_path)
            if saved.is_file():
                word_vectors = np.load(saved_path)
            else:
                word_vectors = self.train_we(pos_src, neg_src)
        self.X, self.Y, self.testX = average_vector(
            pos_src=pos_src,
            neg_src=neg_src,
            test_src=test_src,
            embedding=word_vectors)

    def train_we(self, pos_src, neg_src):
        import subprocess
        subprocess.call(["bash", "-c",
                         r'cat %s %s | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../output/datasources/GloVe/vocab.txt' % (pos_src, neg_src)])
        subprocess.call(["bash", "-c",
                         r'cat ../output/datasources/GloVe/vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d" " -f2 > ../output/datasources/GloVe/vocab_cut.txt'])
        vocab = dict()
        with open('../output/datasources/GloVe/vocab_cut.txt') as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx
        vocab_size = len(vocab)

        data, row, col = [], [], []
        counter = 1
        for fn in ['../../data/train_pos.txt', '../../data/train_neg.txt']:
            with open(fn) as f:
                for line in f:
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]
                    for t in tokens:
                        for t2 in tokens:
                            data.append(1)
                            row.append(t)
                            col.append(t2)

                    if counter % 10000 == 0:
                        print(counter)
                    counter += 1
        cooc = coo_matrix((data, (row, col)))
        print("summing duplicates (this can take a while)")
        cooc.sum_duplicates()
        # with open('cooc.pkl', 'wb') as f:
        #     pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        # print("loading cooccurrence matrix")
        # with open('cooc.pkl', 'rb') as f:
        #     cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        print("using nmax =", nmax, ", cooc.max() =", cooc.max())

        print("initializing embeddings")
        print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
        embedding_dim = 20
        xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
        ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

        eta = 0.001
        alpha = 3 / 4

        epochs = 20

        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
                logn = np.log(n)
                fn = min(1.0, (n / nmax) ** alpha)
                x, y = xs[ix, :], ys[jy, :]
                scale = 2 * eta * fn * (logn - np.dot(x, y))
                xs[ix, :] += scale * y
                ys[jy, :] += scale * x

        we = xs + ys
        with open("../output/datasources/GloVe/glove_word_embedding.txt", "w") as f, open("../output/datasources/GloVe/vocab_cut.txt", "r") as g:
            f.write(str(cooc.shape[0]) + " " + str(embedding_dim) + "\n")
            for i, word in enumerate(g):
                coords = ' '.join([str(b) for b in we[i].tolist()])
                f.write(word.strip() + " " + coords + "\n")
        np.savez('embeddings.npz', we=we)

        return we

