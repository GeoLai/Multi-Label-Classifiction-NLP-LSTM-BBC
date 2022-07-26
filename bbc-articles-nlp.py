# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:12:30 2022

@author: Lai Kar Wei
"""

#%%
# Importing modules
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Constants
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')

#%% Step 1 Data loading
df = pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")

#%% Step 2 Data Inspection

df.info() # info of the dataframe
df.describe().T #stats summary of the dataframe
df.head() # first 5 observation entries

df.isna().sum() # checking for null values
df.duplicated().sum() # checking for repeating entry
df = df.drop_duplicates() # drop all repeating entries
df.shape # getting the latest shape of the dataframe

# read random entry in the dataframe for what type of irregularities
print(df['text'][5])
print(df['text'][500])

#%% Step 3 Data Cleaning
cat = df['category']
txt = df['text']

#removing special characters, whitespaces, letter case
# for index, text in enumerate(txt):
#     txt[index] = re.sub(r'[^a-zA-Z0-9]', ' ', str(text)) # remove punctuation
#     txt[index] = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower() # remove irrelevant char, lowercase
#     txt[index] = re.sub(r'^\s*|\s\s*', ' ', str(text)).strip().split() # remove whitespaces

# print(txt[84]) # test after the for loop
# print(txt[83]) # test after the for loop
# print(txt[85]) # test after the for loop
# print(txt[2214]) # test after the for loop

nb_class = len(cat.unique()) # get amount of class  in 'category' for n-class in training output
print(nb_class)
#%% Step 4  Features Selection
#%% Step 5 Data Preprocessing

#X features
vocab_size = 10000 # set nums of words
oov_token = '<OOV>' #out of vocabulary

tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token=oov_token)

tokenizer.fit_on_texts(txt)
txt_int = tokenizer.texts_to_sequences(txt)
word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

max_len = np.median([len(txt_int[i]) for i in range(len(txt_int))])
print(max_len)

#add padding to the array
X = pad_sequences(txt_int, maxlen=int(max_len), 
                  padding='post', truncating='post')

#Y target
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(cat, axis=-1))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123)

#%% Model Development
from bbc_nlp_module import ModDev

md = ModDev()
model = md.dl_model(X_train, nb_class, vocab_size=10000, out_dims=128)

plot_model(model, show_shapes=(True))

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['acc'])

#%% Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=5,verbose=1, callbacks=[tensorboard_callback])

#%% Model Analysis

y_pred = np.argmax(model.predict(X_test), axis=1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual, y_pred))

#%%
#Tokenizer
token_json = tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH, 'w') as file:
    json.dump(token_json, file)

#OHE
with open(OHE_SAVE_PATH, 'wb') as file:
    pickle.dump(ohe, file)

#Model
model.save(MODEL_SAVE_PATH)
