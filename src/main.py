#!/usr/bin/env python3
import os
import sys
import models
import datetime
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

pos_default = "../data/twitter-datasets/train_pos.txt"
neg_default = "../data/twitter-datasets/train_neg.txt"
test_default = "../data/twitter-datasets/test_data.txt"
embedding_file = "./datasources/word2vec_embedding.txt"

if __name__ == "__main__":
    if not os.getcwd().endswith("/src"):
        os.chdir("src")

    argc = len(sys.argv)

    # print usage
    if argc < 2 or sys.argv[1] not in models.__all__:
        print("Usage: python3.6 main.py [model] [path1] [path2] [path3]")
        print("    Available models: " + str(models.__all__))
        print("    Path to positive training dataset (path1): default is " + pos_default)
        print("    Path to negative training dataset (path2): default is " + neg_default)
        print("    Path to test dataset (path3): default is " + test_default)
    else:
        pos_src = pos_default if argc < 3 else sys.argv[2]
        neg_src = neg_default if argc < 4 else sys.argv[3]
        test_src = test_default if argc < 5 else sys.argv[4]
        
        ### general pipeline here 
        # 1. generate word embedding
        word_vectors = KeyedVectors.load_word2vec_format(embedding_file)
        
        # 2. generate document embedding and build the label vector
        X = average_vector(text_files=[pos_default, neg_default], embedding=word_vectors)   # need to add import for this
        y = np.array([])   
        for fn in [pos_default, neg_default]:
            with open(fn) as f:
                lines = f.readlines()
                n_lines = len(lines)
                if fn == pos_default:
                    y = np.full((n_lines, 1), 1)
                else:
                    labels = np.full((n_lines, 1), -1)
                    y = np.vstack((y, labels))
        
        # 3. split dataset to training set and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

        # Train the model
        print("Training " + sys.argv[1] + "model on positive dataset " + pos_src +
              " and negative dataset " + neg_src)
        model = models.getModel(sys.argv[1])()
        model.train(pos_src, neg_src)

        # 4. get and output the validation error

        # Evaluate for Kaggle submission
        print("Evaluating for kaggle submission on test set " + test_src)
        utils.ensure_dir("../output/models/%s" % sys.argv[1])
        model.evaluate_for_kaggle(test_src,
                                  "../output/models/%s/kaggle_%s.csv" % (sys.argv[1], datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
