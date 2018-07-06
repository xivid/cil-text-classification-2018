from core import BaseModel
from sklearn.utils.validation import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger("LogisticRegression")

class LR(BaseModel):
    def init(self, penalty=1.0, valid_size=0.33):
        self.valid_size = valid_size
        self.model = LogisticRegression(C=penalty)
    
    def train(self):
        logger.info("Fitting Logistic Regression model...")

        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size, random_state=42)

        self.model = self.model.fit(X_train, y_train)
        logger.info("Trained model: " + str(self.model))

        # save the trained model
        # logger.info("Saving the model to " + self.save_path)
        # pickle.dump(self.model, self.save_path)

        # calculate errors
        train_pred = self.predict(X_train)
        self.training_accuracy = accuracy_score(y_train, train_pred)
        logger.info("Training accuracy: %f" % self.training_accuracy)

        val_pred = self.predict(X_val)
        self.validation_accuracy = accuracy_score(y_val, val_pred)
        logger.info("Validating accuracy: %f" % self.validation_accuracy)

    def predict(self, X):
        X = check_array(X)
        pred = self.model.predict(X)
        return pred

