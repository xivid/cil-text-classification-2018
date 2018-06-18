from gensim.models import Word2Vec

pos_filename = '../../data/train_pos.txt'
neg_filename = '../../data/train_neg.txt'
test_filename = '../../data/test_data_stripped.txt'

sentences = []
maxLen = 0;
for fn in [pos_filename, neg_filename, test_filename]:
    with open(fn) as f:
        for line in f:
            splitted = line.split()
            if maxLen < len(splitted):
                maxLen = len(splitted)
            sentences.append(splitted)

print("Starting word2vec")
print(maxLen)
feature_dimensions = 100
word_model = Word2Vec(sentences, size=feature_dimensions)
word_model.wv.save_word2vec_format("word2vec_embedding.txt")
