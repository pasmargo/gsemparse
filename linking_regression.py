
import argparse
import codecs
from collections import Counter
import json
import logging
import numpy as np
import random
import re
import sys
import os

from keras.models import Model
from keras.layers import Embedding, Flatten, UpSampling1D, Reshape
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Dot
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import tensorflow as tf

from models import make_decoder
from models import make_encoder
from models import make_label_input
from models import make_siamese_model
from models import make_siamese_model2
from preprocessing import char_indices
from preprocessing import labels_to_matrix
from preprocessing import load_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", nargs='?', default="char-cnn",
        choices=["char-cnn", "char-lstm"])
    parser.add_argument("--loss_type", nargs='?', default="i1i1",
        choices=["i1i1", "i1i1-i1i2s", "i1i1-i1i2s-i1j1s"])
    parser.add_argument("--ntrain", nargs='?', type=int, default=-1)
    parser.add_argument("--maxlen", nargs='?', type=int, default=16,
        help="Maximum length of labels. Longer labels are cropped.")
    parser.add_argument("--char_emb_size", nargs='?', type=int, default=128)
    parser.add_argument("--batch_size", nargs='?', type=int, default=500)
    parser.add_argument("--epochs", nargs='?', type=int, default=200)
    parser.add_argument("--patience", nargs='?', type=int, default=5)
    parser.add_argument("--exp_suffix", nargs='?', default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    model_basename = '{0}_linkreg{1}'.format(
        args.model_type, args.exp_suffix)
    model_name = model_basename

    labels = load_labels('data/dbpedia_ents.text.jsonl', ntrain=args.ntrain)
    labels_len = [len(label) for label in labels]
    logging.info('Label length stats. min, avg, median, max: {0}, {1}, {2}, {3}'.format(
        np.min(labels_len), np.mean(labels_len), np.median(labels_len), np.max(labels_len)))
    logging.info('Label samples: {0}'.format(random.sample(labels, 10)))
    logging.info('Using {0} characters: {1}'.format(len(char_indices), ''.join(list(char_indices.keys()))))
    char_counter = Counter(c for label in labels for c in label if c not in char_indices)
    logging.info('Unique discarded characters: {0}.'.format(len(char_counter)))
    logging.info('Most common discarded chars: {0}'.format(
        char_counter.most_common(10)))

    X = labels_to_matrix(labels, args.maxlen)
    print('Sample X:\n{}'.format(X[:2, :]))

    # shuffle
    ids = np.arange(len(X))
    np.random.shuffle(ids)
    X = X[ids]

    split = int(len(labels) * .8)
    X_train = X[:split]
    X_test = X[split:]
    logging.info('Using {0} data for train, {1} data for validation.'.format(len(X_train), len(X_test)))    
    
    inputs, outputs, char_emb_x = make_encoder(
        maxlen=args.maxlen,
        char_emb_size=args.char_emb_size)
    encoder = Model(inputs=inputs, outputs=outputs, name=model_basename + '_encoder')
    encoder.summary()

    input1 = make_label_input(args.maxlen)
    input2 = make_label_input(args.maxlen)
    out1, out2 = make_siamese_model2(input1, input2, encoder, drop=0.3)
    emb1 = Model(inputs=[input1], outputs=[out1], name='emb1')
    emb2 = Model(inputs=[input2], outputs=[out2], name='emb2')

    out = Dot(axes=(1,1), normalize=True, name='siam_cos_simil')([out1, out2])
    model = Model(inputs=[input1, input2], outputs=[out], name='siamese_simil')
    optimizer = 'rmsprop'
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    y_train = np.ones((X_train.shape[0], 1), dtype='float32')
    y_test = np.ones((X_test.shape[0], 1), dtype='float32')

    ids_shuffled = np.arange(len(X_train))
    np.random.shuffle(ids_shuffled)
    X_train_shuffled = X_train[ids_shuffled]
    y_train_neg = np.zeros((X_train_shuffled.shape[0], 1), dtype='float32')

    ids_shuffled = np.arange(len(X_test))
    np.random.shuffle(ids_shuffled)
    X_test_shuffled = X_test[ids_shuffled]
    y_test_neg = np.zeros((X_test_shuffled.shape[0], 1), dtype='float32')

    X_train_input1 = np.vstack([X_train, X_train])
    X_train_input2 = np.vstack([X_train, X_train_shuffled])
    y_train = np.vstack([y_train, y_train_neg])

    X_test_input1 = np.vstack([X_test, X_test])
    X_test_input2 = np.vstack([X_test, X_test_shuffled])
    y_test = np.vstack([y_test, y_test_neg])

    ids = np.arange(len(X_train_input1))
    np.random.shuffle(ids)
    X_train_input1 = X_train_input1[ids]
    X_train_input2 = X_train_input2[ids]
    y_train = y_train[ids]

    ids = np.arange(len(X_test_input1))
    np.random.shuffle(ids)
    X_test_input1 = X_test_input1[ids]
    X_test_input2 = X_test_input2[ids]
    y_test = y_test[ids]

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0, mode='auto'),
        CSVLogger(filename=model_name + '.log.csv'),
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint(model_name + '.check',
                        save_best_only=True,
                        save_weights_only=True)]

    model.fit(
        [X_train_input1, X_train_input2],
        y_train,
        validation_data=([X_test_input1, X_test_input2], y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks)
    model.save(model_name + '.h5')
    print('Model wrote to {0}.h5'.format(model_name))
    emb1.save(model_name + '_emb1.h5')
    emb2.save(model_name + '_emb2.h5')
    print('Embedding model 1 wrote to {0}.h5'.format(model_name + '_emb1'))
    print('Embedding model 2 wrote to {0}.h5'.format(model_name + '_emb2'))
