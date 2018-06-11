#!/usr/bin/env python3
import sys
import models
import datetime

pos_default = "../data/twitter-datasets/train_pos_full.txt"
neg_default = "../data/twitter-datasets/train_neg_full.txt"
test_default = "../data/twitter-datasets/test_data.txt"

if __name__ == "__main__":
    argc = len(sys.argv)

    # print usage
    if argc < 2 or sys.argv[1] not in models.__all__ or (argc > 2 and sys.argv[2] not in ("train", "predict")):
        print("Usage: python3.6 main.py [model] [action] [path1] [path2] [path3]")
        print("    Available models: " + str(models.__all__))
        print("    Actions: [\"train\" (default), \"predict\"]")
        print("    Path to positive dataset (path1, training only): default is " + pos_default)
        print("    Path to negative dataset (path2, training only): default is " + neg_default)
        print("    Path to test dataset (path1 for predicting, path3 for training): default is " + test_default)
    else:
        # predict using the model trained last time (if exists)
        if len(sys.argv) > 2 and sys.argv[2] == "predict":
            test_src = test_default if argc < 3 else sys.argv[3]
            print("Predicting with the last trained" + sys.argv[1] + " model on dataset " + test_src)
        # train a new model, save it and predict with test data
        else:
            pos_src = pos_default if argc < 3 else sys.argv[3]
            neg_src = neg_default if argc < 4 else sys.argv[4]
            test_src = test_default if argc < 5 else sys.argv[5]
            print("Training " + sys.argv[1] + "model on positive dataset " + pos_src +
                  " and negative dataset " + neg_src)
            model = models.getModel(sys.argv[1])()
            model.train(pos_src, neg_src)
            model.evaluate_for_kaggle(test_src,
                                      "../output/kaggle_%s.csv" % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
