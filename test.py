
import codecs
import json
import logging
import numpy as np
import random
import re
import sys
import os

from keras.models import Model

from models import make_encoder
from models import make_decoder
from models import make_label_input
from models import make_siamese_model
from preprocessing import char_indices
from preprocessing import labels_to_matrix

logging.basicConfig(level=logging.INFO)

labels1 = ['angelina jolie', 'tom cruise']
labels2 = ['angelina joly', 'tom cruse']
# labels = ['oro verde (argentina)', 'mericiless parlaiment']
# from pudb import set_trace; set_trace()
X1 = labels_to_matrix(labels1)
X2 = labels_to_matrix(labels2)
print('Sample X:\n{}'.format(X1))

maxlen = 16
char_emb_size = 128
num_filters = (char_emb_size, char_emb_size * 2, char_emb_size * 4)
filter_lengths = (3, 3, 3)
subsamples = (1, 1, 1)
pool_lengths = (2, 2, 2)
model_fname = 'char-cnn_linkent_i1i1-i1i2s.check'

inputs, outputs, char_emb_x = make_encoder(
    maxlen,
    char_emb_size,
    num_filters=num_filters,
    filter_lengths=filter_lengths,
    subsamples=subsamples,
    pool_lengths=pool_lengths)
encoder = Model(inputs=inputs, outputs=outputs)
encoder.load_weights(model_fname, by_name=True)

input1 = make_label_input(maxlen)
input2 = make_label_input(maxlen)
siamese = make_siamese_model(input1, input2, encoder, drop=0.0)
siamese.load_weights(model_fname, by_name=True)

simil = siamese.predict([X1, X1])
print(simil)
simil = siamese.predict([X1, X2])
print(simil)
print()

encoded_x1 = encoder.predict(X1)
print(encoded_x1)
encoded_x2 = encoder.predict(X2)
print(encoded_x2)
print()
import sklearn.metrics.pairwise as pairwise
print(1- pairwise.pairwise_distances(encoded_x1[1:], encoded_x1, metric='cosine', n_jobs=-1))
print(1- pairwise.pairwise_distances(encoded_x1[1:], encoded_x2, metric='cosine', n_jobs=-1))
print()
print(pairwise.cosine_similarity(encoded_x1, encoded_x1))
print(pairwise.cosine_similarity(encoded_x1, encoded_x2))

