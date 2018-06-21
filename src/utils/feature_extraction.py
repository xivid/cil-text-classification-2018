#!/usr/bin/env python3
from scipy.sparse import csr_matrix
import numpy as np
import logging

pos_src = '../../data/train_pos.txt'
neg_src = '../../data/train_neg.txt'

def main():
    data = np.load('embeddings.npz') 
    we = data['we']
    print(we)
    
    word_embedding = {}
    #load we and assign vector to each word
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            word_embedding[line.rstrip()] = we[idx]
    print(word_embedding)
    
    avg = []
    counter = 0
    for fn in ['../../data/train_pos.txt', '../../data/train_neg.txt']:
        with open(fn) as f:
            for line in f:
                tokens = line.split()
                avg.append(np.zeros(20))

                num_tokens = 0
                for t in tokens:
                    if((len(t) != 1) or (t == "a") or (t == "i")):
                        try:
                            avg[counter] += word_embedding[t]
                            num_tokens += 1
                        except:
                            continue
                avg[counter] /= num_tokens
                counter += 1

# Adapted from bayes.py 
def sparse_matrix(text_files=[pos_src, neg_src]):
    # gather information for sparse matrix\
    vocab = dict()

    indptr = [0]
    indices = []
    data = []
    for fn in text_files:
        with open(fn) as f:
            lines = f.readlines()
            n_lines = len(lines)
            cnt = 0
            
            for line in lines:
                tokens = line.strip().split()
                for token in tokens:
                    index = vocab.setdefault(token, len(vocab))
                    indices.append(index)
                    data.append(1)
                indptr.append(len(indices))
                cnt += 1
                if cnt % 10000 == 0:
                    print("%d/%d loaded" % (cnt, n_lines))
    
    X = csr_matrix((data, indices, indptr), dtype=int)
    
    return X


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

if __name__ == '__main__':
    main()
