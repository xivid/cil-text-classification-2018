import gensim
import os
import collections
import smart_open
import random

def read_corpus(fname, tokens_only=False):
    print("reading corpus")
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def main():
    pos_filename = '../../data/train_pos.txt'
    neg_filename = '../../data/train_neg.txt'
    test_filename = '../../data/test_data_stripped.txt'

    max_epochs = 10
    alpha = 0.025

    train_corpus = list(read_corpus(pos_filename))
    print(train_corpus)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=55)
    model.build_vocab(train_corpus)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("d2v.model")
#    load the word2vec
#    model = gensim.models.Doc2Vec.load('d2v.model')
#    docvecs = model.docvecs
#    print (docvecs[3])

if __name__ == "__main__":
    main()

