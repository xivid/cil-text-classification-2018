from core import BaseModel
import tensorflow as tf
from sklearn.utils.validation import check_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from keras.models import Sequential, Model

from keras.layers import LSTM, Dropout, Activation, Dense, Embedding

from keras import optimizers

logger = logging.getLogger("LSTM")

class LSTMModel(BaseModel):
    def __init__(self, data_source, save_path=None, kernel='linear', penalty=1.0, valid_size=0.33):
        BaseModel.__init__(self, data_source, save_path)
        self.valid_size = valid_size
        self.model = None
    
    
    def train(self):
        logger.info("Fitting LSTM model...")

        
        embedding_dim = 300
        sequence_length = 100

        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size, random_state=42)
        print(X_train.shape)
        max_features = 1024

        model = Sequential()
        print("sequential")
        model.add(Embedding(max_features, output_dim=400))
        print("embedding")

        model.add(LSTM(128))
        print("lstm ")

        model.add(Dropout(0.5))
        print("dropout ")

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        print(model.summary())

        model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1, shuffle=True)

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
        pred = self.model.predict(X, batch_size=128, verbose=1)
        return pred

