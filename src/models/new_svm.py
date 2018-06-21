from core import BaseModel
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("SVM")

class SVM(BaseModel):
    def __init__(self, data_source, save_path=None, kernel='linear', penalty=1.0, valid_size=0.33):
        BaseModel.__init__(self, data_source, save_path)
        self.valid_size = valid_size
        self.model = SVC(C=penalty,
                         kernel=kernel,
                         class_weight="balanced")
    
    def train(self, X, y):
        logger.info("Start fitting SVM model...")
        X = check_array(X)
        y = check_array(y)

        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size, random_state=42)

        self.model = self.model.fit(X_train, y_train)
        logger.info("Trained model: " + str(self.model))

        # TODO: save errors
        self.training_error = 0.0
        self.validation_error = 0.0
    
    def predict(self, X):
        X = check_array(X)
        pred = self.model.predict(X)
        return pred

