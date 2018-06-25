#!/bin/bash
echo "===================CIL Text Classification====================="
echo "Warning: if you have not connected to the ETH VPN, please do so before running this script!"
echo "===================Downloading dataset========================="

# Move to where this script is located
cd "$(dirname $0)"

echo "Downloading twitter dataset..."
wget -N "http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip"
unzip -o twitter-datasets.zip

echo "Downloading Pre-Trained GloVe Word Embeddings for Twitter..."
wget -N http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -o glove.twitter.27B.zip -d glove.twitter.27B

echo "Converting GloVe to Word2Vec..."
python3 glove_to_word2vec.py glove.twitter.27B/glove.twitter.27B.50d.txt glove.twitter.27B/glove.twitter.27B.50d.word2vec.txt
python3 glove_to_word2vec.py glove.twitter.27B/glove.twitter.27B.100d.txt glove.twitter.27B/glove.twitter.27B.100d.word2vec.txt
python3 glove_to_word2vec.py glove.twitter.27B/glove.twitter.27B.200d.txt glove.twitter.27B/glove.twitter.27B.200d.word2vec.txt

# Move back
cd -
