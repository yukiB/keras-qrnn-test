from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Input
from keras.models import Model
from keras.layers.recurrent import LSTM
from qrnn import QRNN
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

in_out_neurons = 1
hidden_neurons = 64
f_model = "./model"
f_img = "./img"


class SequentialModel():

    def __init__(self, model_type, data_type, l_seq, n_prediction):
        self.old_session = KTF.get_session()
        self.session = tf.Session('')
        KTF.set_session(self.session)
        self.model_type = model_type
        self.data_type = data_type
        self.n_prediction = str(n_prediction)
        if model_type == "cnn":
            self.model = create_cnn_model(l_seq)
        elif model_type == "rnn":
            self.model = create_rnn_model(l_seq)
        else:
            self.model = create_qrnn_model(l_seq)

    def save(self):
        json_string = self.model.to_json()
        open(os.path.join(f_model, self.model_type + '_' + self.data_type +
                          '_pred' + self.n_prediction + '.json'), 'w').write(json_string)
        print("saved... " + self.model_type + '_' + self.data_type + '_pred' + self.n_prediction + '.json')
        self.model.save_weights(os.path.join(f_model, self.model_type + '_' +
                                             self.data_type + '_pred' + self.n_prediction + '_weights.hdf5'))
        print("saved... " + self.model_type + '_' + self.data_type +
              '_pred' + self.n_prediction + '_weights.hdf5')

    def train(self, X_train, y_train, X_test, y_test, epoch):
        self.model.fit(X_train[0:5000],  y_train[0:5000],
                       batch_size=20,
                       nb_epoch=epoch,
                       validation_data=(X_test[0:500], y_test[0:500]))

    def predict(self, X_test, y_test):
        predicted_5_ahead = self.model.predict(X_test)
        dataf = pd.DataFrame(predicted_5_ahead[0:300])
        dataf.columns = ["predict"]
        dataf["true_value(observed_value)"] = y_test[0:300]
        dataf.plot()
        plt.savefig(os.path.join(f_img, self.model_type + '_' +
                                 self.data_type + '_pred' + self.n_prediction + '.png'))

    def sequential_predict(self, dataf, l_seq, start=0):
        l_pred = 350
        now = dataf.iloc[start:start + l_seq].as_matrix()
        df = pd.DataFrame(dataf.iloc[start + l_seq - 150: start + l_seq + l_pred].as_matrix())
        df.columns = ["true_value(observed_value)"]
        pred = []
        for i in range(l_pred):
            p = self.model.predict(np.array([now]))
            pred.append(p[0][0])
            now = np.roll(now, -1)
            now[-1] = pred[-1]
        df["predict"] = [None] * 150 + pred
        df.plot()
        plt.savefig(os.path.join(f_img, self.model_type + '_' +
                                 self.data_type + '_pred' + self.n_prediction + '_2.png'))

    def load(self):
        print("load... " + self.model_type + '_' + self.data_type +
              '_pred' + self.n_prediction + '_weights.hdf5')
        self.model.load_weights(os.path.join(f_model, self.model_type + '_' +
                                             self.data_type + '_pred' + self.n_prediction + '_weights.hdf5'))


def create_cnn_model(l_seq):
    inputs = Input(shape=(l_seq, in_out_neurons))
    x = Conv1D(32,  3, activation='relu', padding='valid')(inputs)
    x = Conv1D(32,  3, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64,  3, activation='relu', padding='valid')(inputs)
    x = Conv1D(64,  3, activation='relu', padding='valid')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(in_out_neurons, activation='linear')(x)
    model = Model(input=inputs, output=predictions)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def create_rnn_model(l_seq):
    inputs = Input(shape=(l_seq, in_out_neurons,))
    x = LSTM(hidden_neurons, return_sequences=False)(inputs)
    predictions = Dense(in_out_neurons, activation='linear')(x)
    model = Model(input=inputs, output=predictions)
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def create_qrnn_model(l_seq):
    input_layer = Input(shape=(l_seq, 1))
    qrnn_output_layer = QRNN(64, window_size=60, dropout=0)(input_layer)
    prediction_result = Dense(1)(qrnn_output_layer)
    model = Model(input=input_layer, output=prediction_result)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
