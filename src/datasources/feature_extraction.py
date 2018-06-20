#!/usr/bin/env python3
from scipy.sparse import csr_matrix
import numpy as np
import pickle
import random

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

def average_vector(text_files=[pos_src, neg_src], embedding): # embedding: KeyedVectors type
    X = np.array([])
    for f in text_files:
        with open(fn) as f:
            lines = f.readlines()
            num_tweet = len(lines)
            average_vectors = np.zeros((num_tweet, embedding.vector_size), dtype=np.float32)
            # build average vector for each line
            for idx, line in enumerate(lines):
                tokens = line.split()
                num_tokens = len(tokens)
                for token in tokens:
                    if token in embedding.wv.vocab:
                        average_vectors[idx] += embedding.wv[token]
                average_vectors[idx] /= num_tokens
            
            if X.size == 0:
                X = average_vectors
            else:
                X = np.vstack((X, average_vectors))
                
    return X

if __name__ == '__main__':
    main()

