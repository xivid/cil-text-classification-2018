from core import BaseModel
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


class SVM(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

        """ Word embedding
        """
        self.word_embedding = None

        """ X
            GloVe average vector
        """
        self.X = None

        """ Y
            Class labels. 1 for positive, -1 for negative.
        """
        self.Y = None

        """ model
            Trained model.
        """
        self.model = None

    def mean_vector(self, pos_src, neg_src):
        data = np.load('embeddings.npz') 
        we = data['we']
        print(we)

        self.word_embedding = {}
        #load we and assign vector to each word
        with open('vocab_cut.txt') as f:
            for idx, line in enumerate(f):
                self.word_embedding[line.rstrip()] = we[idx]
        #print(word_embedding)

        avg = []
        y = []
        counter = 0
        is_positive = 1
        for fn in [pos_src, neg_src]:
            with open(fn) as f:
                n_lines = 0
                for line in f:
                    n_lines += 1
                    tokens = line.split();
                    avg.append(np.zeros(20))

                    num_tokens = 0;
                    for t in tokens:
                        try:
                            avg[counter] += self.word_embedding[t]
                            num_tokens += 1;
                        except:
                            continue;
                    if (num_tokens != 0):
                        np.true_divide(avg[counter], num_tokens)
                    counter += 1
                y += n_lines * [is_positive]
            is_positive -= 2
        print("retunring values")
        return {'x':avg, 'y':y} 

    def load_training_data(self, pos_src, neg_src):
        print("loading cooccurrence matrix")
        """with open('cooc.pkl', 'rb') as f:
            cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        print("using nmax =", nmax, ", cooc.max() =", cooc.max())

        print("initializing embeddings");
        print("cooc shape 0: ", cooc.shape[0], "cooc shape 1: ", cooc.shape[1])
        embedding_dim = 20
        xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
        ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

        eta = 0.001
        alpha = 3 / 4

        epochs = 20

        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
                logn = np.log(n)
                fn = min(1.0, (n / nmax) ** alpha)
                x, y = xs[ix, :], ys[jy, :]
                scale = 2 * eta * fn * (logn - np.dot(x, y))
                xs[ix, :] += scale * y
                ys[jy, :] += scale * x

        we = xs + ys
        np.savez('embeddings', we=we)"""
        

        # build sparse matrix
        result = self.mean_vector(pos_src, neg_src)
        print("get result")
        self.X = result['x']
        self.Y = np.array(result['y'])
    
    def train(self, pos_src, neg_src):
        self.load_training_data(pos_src, neg_src)
        print("train")
        clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), n_jobs=4)
        print("svm create")
      
        clf.fit(self.X, self.Y) 
      

        print("svm fit")
       

        print("Trained model: " + str(self.model))

    def predict(self, test_src):
        if self.model is None:
            raise RuntimeError("No trained model! (saving model not implemented yet)")
        print("loading test data...")
        avg = []
        counter = 0
        with open(test_src) as f:
            counter = 0
            for line in f:
                tokens = line.split();
                avg.append(np.zeros(20))

                num_tokens = 0;
                for t in tokens:
                    if((len(t) != 1) or (t == "a") or (t == "i")):
                        try:
                            avg[counter] += self.word_embedding[t]
                            num_tokens += 1;
                        except:
                            continue;
                np.divide(avg[counter], num_tokens)
                counter += 1

        return self.model.predict(avg)
