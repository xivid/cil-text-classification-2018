#!/usr/bin/env python3
from core import BaseDataSource
from scipy.sparse import *
import numpy as np
from utils.feature_extraction import average_vector
from gensim.models import KeyedVectors
from pathlib import Path
import logging

logger = logging.getLogger("GloVe")
output_dir = "../output/datasources/GloVe/"


class GloVe(BaseDataSource):
    """
    This GloVe class train the word embeddings in preprocess()->train_we() (and output the intermediate results to ../../output/datasources/GloVe/),
    and yield (the feature vector and the corresponding class label of) one sample on each call to yield_one_sample(),
    or expose all samples with self.X, self.Y.
    """

    def preprocess(self, pos_src, neg_src, test_src, embedding_src=""):
        if embedding_src != "":
            # load pre-trained embeddings
            logger.info("Loading external embeddings")
            word_vectors = KeyedVectors.load_word2vec_format(embedding_src)
        else:
            saved_path = output_dir + "glove_word_embedding.txt"
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
        import subprocess
        from utils import ensure_dir
        ensure_dir(output_dir)

        vocab_path = output_dir + "vocab.txt"
        logger.info("building vocab.txt: " +
                    r'cat %s %s | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > %s'
                    % (pos_src, neg_src, vocab_path))
        subprocess.call(["bash", "-c",
                         r'cat %s %s | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > %s'
                         % (pos_src, neg_src, vocab_path)])

        vocab_cut_path = output_dir + "vocab_cut.txt"
        logger.info("building vocab_cut.txt: " +
                    r'cat %s | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d" " -f2 > %s'
                    % (vocab_path, vocab_cut_path))
        subprocess.call(["bash", "-c",
                         r'cat %s | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d" " -f2 > %s'
                         % (vocab_path, vocab_cut_path)])

        vocab = dict()
        with open(vocab_cut_path) as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx
        vocab_size = len(vocab)

        data, row, col = [], [], []
        counter = 1
        for fn in [pos_src, neg_src]:
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
                        logger.info("read %d samples" % counter)
                    counter += 1
        cooc = coo_matrix((data, (row, col)))
        logger.info("summing duplicates (this can take a while)")
        cooc.sum_duplicates()
        # with open('cooc.pkl', 'wb') as f:
        #     pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        # print("loading cooccurrence matrix")
        # with open('cooc.pkl', 'rb') as f:
        #     cooc = pickle.load(f)
        logger.info("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        logger.info("using nmax = %d, cooc.max() = %s" % (nmax, str(cooc.max())))

        logger.info("initializing embeddings")
        logger.info("cooc shape 0: %s, cooc shape 1: %s" % (str(cooc.shape[0]), str(cooc.shape[1])))
        embedding_dim = 300
        xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
        ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

        eta = 0.001
        alpha = 3 / 4

        epochs = 5

        for epoch in range(epochs):
            logger.info("epoch {}".format(epoch))
            for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
                logn = np.log(n)
                fn = min(1.0, (n / nmax) ** alpha)
                x, y = xs[ix, :], ys[jy, :]
                scale = 2 * eta * fn * (logn - np.dot(x, y))
                xs[ix, :] += scale * y
                ys[jy, :] += scale * x

        we = xs + ys
        with open(save_path, "w") as f, open(vocab_cut_path, "r") as g:
            f.write(str(cooc.shape[0]) + " " + str(embedding_dim) + "\n")
            for i, word in enumerate(g):
                coords = ' '.join([str(b) for b in we[i].tolist()])
                f.write(word.strip() + " " + coords + "\n")

        model = KeyedVectors.load_word2vec_format(save_path)

        return model

