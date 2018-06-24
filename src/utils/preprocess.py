#!/usr/bin/env python3
from scipy.sparse import csr_matrix
import numpy as np
import os
import logging
from gensim.parsing.preprocessing import *

pos_src = '../../data/train_pos.txt'
neg_src = '../../data/train_neg.txt'

def main():
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text]
    
    for fn in [pos_src, neg_src]:
        with open(fn) as f:
            fo = open(os.path.splitext(fn)[0]+"_preprocessed.txt", "w+")
            for line in f:
                str = ' '.join(preprocess_string(line, CUSTOM_FILTERS))
                print(str)
                fo.write(str + "\n")
            fo.close()
if __name__ == '__main__':
    main()

