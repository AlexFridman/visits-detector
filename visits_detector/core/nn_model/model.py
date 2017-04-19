from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

np.random.seed(1337)  # for reproducibility

learning_rate = 0.0001
clipnorm = 15.0
train_keep_prob = 0.75
train_mean = np.array([76.69657889, 0.28964923])
train_std = np.array([52.28199665, 0.67885899])


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def create_model():
    model = Sequential([
        InputLayer(input_shape=(None, 2)),
        Masking(mask_value=-1.),
        Bidirectional(LSTM(32)),
        Dropout(train_keep_prob),
        Dense(1, activation='sigmoid')
    ])

    return model


def compile_model(model):
    optimizer = Adam(lr=learning_rate, clipnorm=clipnorm)

    model.compile(
        optimizer,
        'binary_crossentropy',
        metrics=['accuracy', f1_score]
    )


def normalize(X, mean, std):
    X_normed = []

    for x_i in X:
        X_normed.append((x_i - mean) / std)

    return np.array(X_normed)


class NNModel(object):
    @staticmethod
    def load_model(weights_path):
        model = create_model()
        model.load_weights(weights_path)
        compile_model(model)
        return model

    def __init__(self, weights_path):
        self._keras_model = self.load_model(weights_path)

    def predict(self, x):
        x_normed = normalize(np.array([x]), train_mean, train_std)
        x_padded = pad_sequences(x_normed, 400, padding='post', truncating='post', dtype='float32', value=-1.)
        return self._keras_model.predict_classes(x_padded, verbose=0)[0][0]
