from core import BaseMLModel
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.svm import SVC

class SVM(BaseModel):
    def __init__(self, save_path=None):
        BaseModel.__init__(self, save_path)
        kernel='linear'
        penalty=1.0
        self.model = SVC(C=penalty,
                         kernel=kernel,
                         class_weight="balanced")
    
    def train(self, X, y):
        print("Start fitting SVM model...")
        X = check_array(X)
        y = check_array(y)
        self.model = self.model.fit(X, y)
        print("Trained model: " + str(self.model))
    
    def predict(self, X):
        X = check_array(X)
        pred = self.model.predict(X)
        return pred
        
