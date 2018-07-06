from core import BaseModel
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger("NaiveBayes")

class NaiveBayes(BaseModel):
    def init(self, valid_size=0.33):
        self.model = MultinomialNB()
        self.valid_size = valid_size

    def train(self):
        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size, random_state=42)
        self.model.fit(X_train, y_train)
        print("Trained model: " + str(self.model))

        # calculate errors
        train_pred = self.predict(X_train)
        self.training_accuracy = accuracy_score(y_train, train_pred)
        logger.info("Training accuracy: %f" % self.training_accuracy)

        val_pred = self.predict(X_val)
        self.validation_accuracy = accuracy_score(y_val, val_pred)
        logger.info("Validating accuracy: %f" % self.validation_accuracy)

    def predict(self, X):
        return self.model.predict(X)
