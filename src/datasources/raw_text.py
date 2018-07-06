from core import BaseDataSource
import numpy as np
import logging

logger = logging.getLogger("RawText")


class RawText(BaseDataSource):
    max_tok_count = 0
    embedding_src = ""

    """
    Raw Text
    Returns raw text lines with labels.
    """
    def process(self, pos_src, neg_src, test_src):
        logger.info("Loading training data")
        X = []
        Y = []
        max_tok_count = 0
        for fileno, fn in enumerate([pos_src, neg_src]):
            with open(fn) as f:
                logger.info("processing " + fn)
                lines = f.readlines()
                num_tweet = len(lines)
                counter = 1
                for idx, line in enumerate(lines):
                    tokens = line.split()
                    num_tokens = len(tokens)
                    if num_tokens > max_tok_count:
                        max_tok_count = num_tokens
                    if counter % 10000 == 0:
                        logger.info("read %d samples" % counter)
                    counter += 1
                    X.append(line)
                if fileno == 0:
                    Y = np.array([1] * num_tweet)
                else:
                    Y = np.concatenate((Y, [-1] * num_tweet))

        logger.info("Loading testing data")
        testX = []
        with open(test_src) as f:
            logger.info("processing " + test_src)
            lines = f.readlines()
            num_tweet = len(lines)
            counter = 1
            listX = []
            for idx, line in enumerate(lines):
                tokens = line.split()
                num_tokens = len(tokens)
                if num_tokens > max_tok_count:
                    max_tok_count = num_tokens
                if counter % 10000 == 0:
                    logger.info("read %d samples" % counter)
                counter += 1
                testX.append(line)

        self.X = X
        self.Y = Y
        self.testX = testX
        self.max_tok_count = max_tok_count
