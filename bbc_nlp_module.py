# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:09:45 2022

@author: Lai Kar Wei
"""

import numpy as np

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional

class ModDev:
    def dl_model(self, X_train, nb_class, vocab_size=8000, out_dims=64, nb_node=128, dropout_rate=0.3): 
        #values inside this is default unless stated

        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Embedding(vocab_size, out_dims))
        model.add(Bidirectional(LSTM(128, return_sequences=(True))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation='softmax'))
        model.summary()

        return model

