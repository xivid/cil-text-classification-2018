from core import BaseModel
from sklearn.utils.validation import check_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import lightgbm as lgb

logger = logging.getLogger("LightGBM")


class LightGBM(BaseModel):
    def __init__(self, data_source, save_path=None, valid_size=0.20):
        BaseModel.__init__(self, data_source, save_path)
        self.valid_size = valid_size
        self.model = None

    def train(self):
        logger.info("Fitting LightGBM model...")

        self.data_source.Y[self.data_source.Y == -1] = 0
        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        param = {'num_leaves': 31,
                 'learning_rate': 0.1,
                 'objective': 'binary',
                 'metric': ['binary_error', 'binary_logloss']}
        num_round = 500
        self.model = lgb.train(param, train_data, num_round, valid_sets=[valid_data])
        logger.info("Trained model: " + str(self.model))

        # save the trained model
        # logger.info("Saving the model to " + self.save_path)
        # pickle.dump(self.model, self.save_path)

        # calculate errors
        train_pred = self.predict(X_train)
        y_train[y_train == 0] = -1
        self.training_accuracy = accuracy_score(y_train, train_pred)
        logger.info("Training accuracy: %f" % self.training_accuracy)

        val_pred = self.predict(X_val)
        y_val[y_val == 0] = -1
        self.validation_accuracy = accuracy_score(y_val, val_pred)
        logger.info("Validating accuracy: %f" % self.validation_accuracy)

    def predict(self, X):
        X = check_array(X)
        pred = self.model.predict(X)
        pred[pred < 0.5] = -1
        pred[pred >= 0.5] = 1
        return pred

