
import sys
import logging
from gensim.scripts.glove2word2vec import glove2word2vec

logger = logging.getLogger("Transform")

glove_file = ""
output_file = "" 
output_default = "../data/glove_to_word2vec.txt"

if __name__ == "__main__":
    num_argv = len(sys.argv)
    if(num_argv == 0 or num_argv > 2):
        logger.info("[Error]Invalid parameters!")
    
    if(num_argv == 2):
        glove_file = sys.argv[0]
        output_file = sys.argv[1]

    else:
        glove_file = sys.argv[0]
        output_file = output_default
    
    glove2word2vec(glove_file, output_file)
    logger.info("Successfully transformed GloVe embedding: %s into word2vec format, saved at: %s", 
                 glove_file, output_file)