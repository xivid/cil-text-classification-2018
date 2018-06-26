import os
import pathlib


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def file_type(path):   # return the type of embedding file
    logger.info("Loading external embeddings")
    file_type = os.path.splitext(embedding_src)[1]
    if file_type == ".bin":
        is_binary = True
    elif file_type == ".txt":
        is_binary = False
    else:
        logger.error("Unsupported file type!")
    return is_binary
    # word_vectors = KeyedVectors.load_word2vec_format(embedding_src, binary = is_binary)