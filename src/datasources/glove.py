#!/usr/bin/env python3
# from core import BaseDataSource
from scipy.sparse import *
import numpy as np
import pickle
import random

"""
Warning: don't implement a data source like this! Refer to the structure of `BaseDataSource`!

This Glove class needs modifying to do everything (that are done in build_vocab.sh, cooc.py, ...) 
in preprocess() (and output the intermediate results to ../../output/datasources/Glove/),
and yield (the feature vector and the corresponding class label) of one sample on each call to yield_one_sample(). 
"""

# class Glove(BaseDataSource):
class Glove:
    def main(self):
        print("loading cooccurrence matrix")
        with open('cooc.pkl', 'rb') as f:
            cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        print("using nmax =", nmax, ", cooc.max() =", cooc.max())

        print("initializing embeddings");
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
        with open("glove_word_embedding.txt", "w") as f, open("vocab_cut.txt", "r") as g:
            f.write(str(cooc.shape[0]) + " " + str(embedding_dim) + "\n")
            for i, word in enumerate(g):
                coords = ' '.join([str(b) for b in we[i].tolist()])
                f.write(word.strip() + " " + coords + "\n")
        #np.savez('embeddings', we=we)

if __name__ == '__main__':
    Glove().main()
