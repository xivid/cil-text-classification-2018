#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


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
                    
if __name__ == '__main__':
    main()

def 