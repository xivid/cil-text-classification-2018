# CIL Text Classification Project
Goal:  Given a tweet -> decide whether it is positive or negative 

## Task Division
- Word Embedding Algorithm: Ghazal, Chen
- Classifier (CNN architecture):  Yang, Du

### The Example method ( in Exercise 6 ): 
1. Construct word vectors from the tweet using GloVe word embedding algorithm
2. Use average word vector as feature
3. Train a simple linear model ( SVM / Logistic Regression ) 

### Possible improvements
- Feature Selection (e.g. n-gram feature )
      word embedding algorithm / tweet embedding algorithm

- Classifier  (e.g. random forest, ensemble model, CNN, RNN, LSTM ): using deep neural network is popular in recent years 
      ( Yang, Du)

### Recent Literature
1. [Tweet Sentiment Classification](http://nbviewer.jupyter.org/github/dsar/Twitter_Sentiment_Analysis/blob/master/report/report.pdf)    ( a survey )

2. [Leveraging Large Amounts of Weakly Supervised Data for Multi-Language Sentiment Classification](https://arxiv.org/pdf/1703.02504.pdf) 

3. [Learning Sentiment-Specific Word Embedding for Tweet Sentiment Classification](http://www.aclweb.org/anthology/P14-1146) 

4. [Text Sentiment Classification using distant supervision]( https://s3.amazonaws.com/academia.edu.documents/34632156/Twitter_Sentiment_Classification_using_Distant_Supervision.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1525045722&Signature=xSfOztfHPrssgcbCjWmiwlOOQjY%3D&response-content-disposition=inline%3B%20filename%3DTwitter_Sentiment_Classification_using_D.pdf)


### Related Materials
- Existing CNN model:
> https://github.com/abdulfatir/twitter-sentiment-analysis

> https://github.com/danielegrattarola/twitter-sentiment-cnn

> http://ir.hit.edu.cn/~qinb/ (in Chinese)

- Articles
Introduction to Sentiment Analysis
https://nlp.stanford.edu/sentiment/  (Standford NLP Sentiment Analysis)
