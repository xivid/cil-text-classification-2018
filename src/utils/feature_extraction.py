#!/usr/bin/env python3
from scipy.sparse import csr_matrix
import numpy as np
import logging


# build the tweet embedding as the average vector of the word embeddings
def average_vector(pos_src, neg_src, test_src, embedding): # embedding: KeyedVectors type
    logger = logging.getLogger("average_vector")

    X = np.array([])
    for fn in [pos_src, neg_src]:
        with open(fn) as f:
            logger.info("processing " + fn)
            lines = f.readlines()
            num_tweet = len(lines)
            average_vectors = np.zeros((num_tweet, embedding.vector_size), dtype=np.float32)
            # build average vector for each line
            counter = 1
            for idx, line in enumerate(lines):
                tokens = line.split()
                num_tokens = len(tokens)
                for token in tokens:
                    if token in embedding.wv.vocab:
                        average_vectors[idx] += embedding.wv[token]
                if num_tokens != 0:
                    average_vectors[idx] /= num_tokens
                if counter % 10000 == 0:
                    logger.info("read %d samples" % counter)
                counter += 1

            if X.size == 0:
                X = average_vectors
                Y = np.array([1] * num_tweet)
            else:
                X = np.vstack((X, average_vectors))
                Y = np.concatenate((Y, [-1] * num_tweet))

    testX = np.array([])
    with open(test_src) as f:
        logger.info("processing " + test_src)
        lines = f.readlines()
        num_tweet = len(lines)
        average_vectors = np.zeros((num_tweet, embedding.vector_size), dtype=np.float32)
        # build average vector for each line
        counter = 1
        for idx, line in enumerate(lines):
            tokens = line.split()
            num_tokens = len(tokens)
            for token in tokens:
                if token in embedding.wv.vocab:
                    average_vectors[idx] += embedding.wv[token]
            if num_tokens != 0:
                average_vectors[idx] /= num_tokens
            if counter % 10000 == 0:
                logger.info("read %d samples" % counter)
            counter += 1
        testX = average_vectors

    return X, Y, testX


def load_sparse_tuples_from_file(filename, vocabulary, data, indices, indptr):
    logger = logging.getLogger("load_sparse_tuples_from_file")

    f = open(filename, "r")
    lines = f.readlines()
    n_lines = len(lines)
    print("%s: %d samples" % (filename, n_lines))

    cnt = 0
    for line in lines:
        tokens = line.strip().split()
        for token in tokens:
            index = vocabulary.setdefault(token, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
        cnt += 1
        if cnt % 10000 == 0:
            print("%d/%d loaded" % (cnt, n_lines))

    return n_lines