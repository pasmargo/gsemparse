
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
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import tensorflow as tf

from models import make_decoder
from models import make_encoder
from models import make_label_input
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

    model_basename = '{0}_linkent_{1}{2}'.format(
        args.model_type, args.loss_type, args.exp_suffix)
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
    
    #split = int(args.ntrain * .8) # int(-1 * 0.8) = 0
    split = int(len(labels) * .8)
    X_train = X[:split]
    X_test = X[split:]
    logging.info('Using {0} data for train, {1} data for validation.'.format(len(X_train), len(X_test)))    
 
    num_filters = (args.char_emb_size, args.char_emb_size * 2, args.char_emb_size * 4)
    filter_lengths = (3, 3, 3)
    subsamples = (1, 1, 1)
    pool_lengths = (2, 2, 2)
    
    inputs, outputs, char_emb_x = make_encoder(
        args.maxlen,
        args.char_emb_size,
        num_filters=num_filters,
        filter_lengths=filter_lengths,
        subsamples=subsamples,
        pool_lengths=pool_lengths)
    encoder = Model(inputs=inputs, outputs=outputs, name='encoder_model')
    encoder.summary()
    
    char_emb_model = Model(inputs=inputs, outputs=[char_emb_x], name='char_emb_model')
    char_emb_model.summary()
    
    _, outputs = make_decoder(
        outputs[0],
        num_filters=list(reversed(num_filters)),
        filter_lengths=list(reversed(filter_lengths)),
        subsamples=list(reversed(subsamples)),
        up_lengths=list(reversed(pool_lengths)))
    autoencoder = Model(inputs=inputs, outputs=outputs, name='autoencoder_model')
    autoencoder.summary()
    
    loss_out = None
    inputs = []
    if args.loss_type == 'i1i1-i1i2':
        loss_out = Lambda(
            loss_i1i1_i1i2_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(img1_input),
                img1_input,
                encoder(img1_input),
                encoder(img2_input)])
        model = Model(
            inputs=[img1_input, img2_input],
            outputs=[loss_out],
            name=model_name)
    elif args.loss_type == 'i1i1-i1i2s-i1j1s':
        from models import make_siamese_model
        from models import loss_i1i1_i1i2s_i1j1s_func
        input1 = make_label_input(args.maxlen)
        input2 = make_label_input(args.maxlen)
        input3 = make_label_input(args.maxlen)
        inputs = [input1, input2, input3]
        siamese = make_siamese_model(input1, input2, encoder, drop=0.3)
        loss_out = Lambda(
            loss_i1i1_i1i2s_i1j1s_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(input1),
                char_emb_model(input1),
                siamese([input1, input2]),
                siamese([input1, input3])])
        ids = np.arange(len(X_train))
        np.random.shuffle(ids)
        X_train = [X_train, X_train, X_train[ids]]
        ids = np.arange(len(X_test))
        np.random.shuffle(ids)
        X_test = [X_test, X_test, X_test[ids]]
    elif args.loss_type == 'i1i1-i1i2s':
        from models import make_siamese_model
        from models import loss_i1i1_i1i2s_func
        input1 = make_label_input(args.maxlen)
        input2 = make_label_input(args.maxlen)
        inputs = [input1, input2]
        siamese = make_siamese_model(input1, input2, encoder, drop=0.0)
        loss_out = Lambda(
            loss_i1i1_i1i2s_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(input1),
                char_emb_model(input1),
                siamese([input1, input2])])
        X_train = [X_train, X_train]
        X_test = [X_test, X_test]
    elif args.loss_type == 'i1i1':
        from models import loss_i1i1_func as loss_func
        input1 = make_label_input(args.maxlen)
        inputs = [input1]
        loss_out = Lambda(
            loss_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(input1),
                char_emb_model(input1)])
    elif args.loss_type == 'i1i2':
        loss_out = Lambda(
            loss_i1i2_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(img1_input),
                img2_input])
        model = Model(
            inputs=[img1_input, img2_input],
            outputs=[loss_out],
            name=model_name)
    model = Model(
        inputs=inputs,
        output=[loss_out],
        name=model_name)
    
    optimizer = 'rmsprop'
    model.compile(
        optimizer=optimizer,
        loss={'loss_func' : lambda y_true, y_pred: y_pred})
    
    # ReduceLROnPlateau(patience=args.patience / 2, verbose=1),
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0, mode='auto'),
        CSVLogger(filename=model_name + '.log.csv'),
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint(model_name + '.check',
                        save_best_only=True,
                        save_weights_only=True)]
    model.fit(
        X_train,
        X_train[0],
        validation_data=(X_test, X_test[0]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks)

    encoder.save(model_name + '.h5')
    print('Encoder model wrote to {0}.h5'.format(model_name))

