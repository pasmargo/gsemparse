
import codecs
import json
import logging

from keras.models import Model
from keras.layers import Embedding, Flatten, UpSampling1D, Reshape
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import tensorflow as tf

import numpy as np
import random
import re
import sys
import os

from models import make_encoder
from models import make_decoder
from preprocessing import char_indices
from preprocessing import labels_to_matrix
from preprocessing import load_labels

logging.basicConfig(level=logging.INFO)

labels = ['Angelina Jolie', 'Tom Cruise']
X = labels_to_matrix(labels)
print('Sample X:\n{}'.format(X[:2, :]))

maxlen = 16
char_emb_size = 128
num_filters = (char_emb_size, char_emb_size * 2, char_emb_size * 4)
filter_lengths = (3, 3, 3)
subsamples = (1, 1, 1)
pool_lengths = (2, 2, 2)
model_fname = 'char-cnn.check'

inputs, outputs, char_emb_x = make_encoder(
    maxlen,
    char_emb_size,
    num_filters=num_filters,
    filter_lengths=filter_lengths,
    subsamples=subsamples,
    pool_lengths=pool_lengths)
encoder = Model(inputs=inputs, outputs=outputs)
encoder.summary()
encoder.load_weights(model_fname, by_name=True)

encoded = encoder.predict(X)
print(encoded)
print(encoded.shape)
