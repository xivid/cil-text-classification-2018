from core import BaseModel
import tensorflow as tf
from sklearn.utils.validation import check_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM

logger = logging.getLogger("LSTM")

class LSTM(BaseModel):
    def __init__(self, data_source, save_path=None, kernel='linear', penalty=1.0, valid_size=0.33):
        BaseModel.__init__(self, data_source, save_path)
        self.valid_size = valid_size
    
    
    def train(self):
        logger.info("Fitting LSTM model...")
        
        X_train, X_val, y_train, y_val = train_test_split(self.data_source.X, self.data_source.Y, test_size=self.valid_size, random_state=42)
        
        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
        model.add(Dropout(0.4))
        model.add(LSTM(128))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = "./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1, shuffle=True, callbacks=[checkpoint, reduce_lr])
        
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
        pred = model.predict(X, batch_size=128, verbose=1)
        return pred

