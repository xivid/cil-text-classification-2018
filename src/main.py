#!/usr/bin/env python3
import os
import argparse
import coloredlogs
import logging
import models
import datasources
import datetime

logger = logging.getLogger("Main")

pos_default = "../data/twitter-datasets/train_pos.txt"
neg_default = "../data/twitter-datasets/train_neg.txt"
test_default = "../data/twitter-datasets/test_data.txt"
# embedding_default = "../data/embeddings.npz"

if __name__ == "__main__":

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a text classification model.')
    parser.add_argument('-m', '--model', type=str, help='training model', choices=models.__all__, required=True)
    # TODO: add necessary arguments required by models
    # e.g. parser.add_argument('-lr', type=float, help="learning rate", default=0.1)
    parser.add_argument('-d', '--datasource', type=str, help='data source (features used in the model)',
                        choices=datasources.__all__, required=True)
    # TODO: add necessary arguments required by data sources
    parser.add_argument('-p', type=str, help='Path to positive training dataset', default=pos_default)
    parser.add_argument('-n', type=str, help='Path to negative training dataset', default=neg_default)
    parser.add_argument('-t', type=str, help='Path to test dataset', default=test_default)
    parser.add_argument('-e', type=str,
                        help='Path to pre-trained word embeddings (only for GloVe, word2vec and doc2vec data sources)',
                        default="")
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    model_name = args.model
    datasource_name = args.datasource
    pos_src = args.p
    neg_src = args.n
    test_src = args.t
    embedding_src = args.e

    logger.info("Text classification framework launched")
    logger.info("\tusing model: %s", model_name)
    logger.info("\tusing datasource: %s", datasource_name)
    logger.info("\tusing positive training set: %s", pos_src)
    logger.info("\tusing negative training set: %s", neg_src)
    logger.info("\tusing test set: %s", test_src)
    logger.info("\tlogging verbosity: %s", args.v)

    # set working directory
    if not os.getcwd().endswith("/src"):
        os.chdir("src")

    ### general pipeline here
    # maybe do some preprocessing for the raw training data here [TODO]
    # added by YZF: I'd suggest doing such preprocessing (if any) in DataSource, e.g. add some preprocessing methods in BaseDataSource

    # Pre-process data source
    logger.info("Pre-processing data source " + datasource_name)
    data_source = datasources.get_datasource(datasource_name)(
        pos_src=pos_src,
        neg_src=neg_src,
        test_src=test_src,
        embedding_src=embedding_src  # TODO: only pass this parameter when it is really a word embedding data source
    )

    # Train the model
    logger.info("Training " + datasource_name)
    model = models.get_model(model_name)(
        data_source=data_source
    )
    model.train()

    # Output training and validation error
    logging.info("Training finished with error %f, validation error %f" % (model.training_error, model.validation_error))

    # Evaluate for Kaggle submission (need to be modified) TODO
    logging.info("Evaluating for kaggle submission")
    model.evaluate_for_kaggle("../output/models/%s/kaggle_%s.csv" % (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
