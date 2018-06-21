
import sys
import logging
from gensim.scripts.glove2word2vec import glove2word2vec

logger = logging.getLogger("glove_to_word2vec")

glove_file = ""
output_file = "" 

if __name__ == "__main__":
    num_argv = len(sys.argv)
    if num_argv != 3:
        logger.error("Invalid parameters!")
    
    glove_file = sys.argv[1]
    output_file = sys.argv[2]
    logger.info("Converting GloVe file %s to Word2Vec file %s" % (glove_file, output_file))

    glove2word2vec(glove_file, output_file)
    logger.info("Done")