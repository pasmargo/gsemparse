
import codecs
from collections import Counter
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", nargs='?', default="sys",
        choices=["base", "sys"])
    parser.add_argument("--loss_type", nargs='?', default="i1i1-i1i2",
        choices=["i1i1-i1i2", "i1i1-i1i2s", "i1i1", "i1i2"])
    parser.add_argument("--batch_size", nargs='?', type=int, default=40)
    parser.add_argument("--batch_size_val", nargs='?', type=int, default=150)
    parser.add_argument("--epochs", nargs='?', type=int, default=200)
    parser.add_argument("--steps_per_epoch", nargs='?', type=int, default=100)
    parser.add_argument("--validation_steps", nargs='?', type=int, default=1)
    parser.add_argument("--patience", nargs='?', type=int, default=6)
    parser.add_argument("--batch_normalization", nargs='?', type=bool, default=False)
    parser.add_argument("--reduce_params", dest="reduce_params", action="store_true")
    parser.add_argument("--no_reduce_params", dest="reduce_params", action="store_false")
    parser.add_argument("--exp_suffix", nargs='?', default="")
    parser.set_defaults(reduce_params=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    model_basename = 'vgg16_{0}_reduce{1}_{2}{3}'.format(
        args.model_type, str(args.reduce_params), args.loss_type, args.exp_suffix)
    model_name = model_basename + '_multiobj'

    ntrain = 1000000
    maxlen = 16
    char_emb_size = 128
    batch_size = 10000
    max_epochs = 100

    labels = load_labels('dbpedia_ents.text.jsonl', ntrain=ntrain)
    labels_len = [len(label) for label in labels]
    logging.info('Label stats. min, avg, median, max: {0}, {1}, {2}, {3}'.format(
        np.min(labels_len), np.mean(labels_len), np.median(labels_len), np.max(labels_len)))
    logging.info('Label samples: {0}'.format(random.sample(labels, 10)))
    logging.info('Using {0} characters: {1}'.format(len(char_indices), ''.join(list(char_indices.keys()))))
    char_counter = Counter(c for label in labels for c in label if c not in char_indices)
    logging.info('Unique discarded characters: {0}.'.format(len(char_counter)))
    logging.info('Most common discarded chars: {0}'.format(
        char_counter.most_common(10)))

    X = labels_to_matrix(labels, maxlen)
    print('Sample X:\n{}'.format(X[:2, :]))
    
    # shuffle
    ids = np.arange(len(X))
    np.random.shuffle(ids)
    X = X[ids]
    
    split = int(ntrain * .8)
    X_train = X[:split]
    X_test = X[split:]
    
    num_filters = (char_emb_size, char_emb_size * 2, char_emb_size * 4)
    filter_lengths = (3, 3, 3)
    subsamples = (1, 1, 1)
    pool_lengths = (2, 2, 2)
    
    inputs, outputs, char_emb_x = make_encoder(
        maxlen,
        char_emb_size,
        num_filters=num_filters,
        filter_lengths=filter_lengths,
        subsamples=subsamples,
        pool_lengths=pool_lengths)
    encoder = Model(inputs=inputs, outputs=outputs)
    encoder.summary()
    
    char_emb_model = Model(inputs=inputs, outputs=[char_emb_x])
    char_emb_model.summary()
    
    _, outputs = make_decoder(
        outputs[0],
        num_filters=list(reversed(num_filters)),
        filter_lengths=list(reversed(filter_lengths)),
        subsamples=list(reversed(subsamples)),
        up_lengths=list(reversed(pool_lengths)))
    autoencoder = Model(inputs=inputs, outputs=outputs)
    autoencoder.summary()
    
    img1_input = Input(shape=(224, 224, 3))
    img2_input = Input(shape=(224, 224, 3))
    
    loss_out = None
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
    elif args.loss_type == 'i1i1-i1i2s':
        siamese = make_siamese_model(img1_input, img2_input, encoder)
        loss_out = Lambda(
            loss_i1i1_i1i2s_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(img1_input),
                img1_input,
                siamese([img1_input, img2_input])])
        model = Model(
            inputs=[img1_input, img2_input],
            outputs=[loss_out],
            name=model_name)
    elif args.loss_type == 'i1i1':
        loss_out = Lambda(
            loss_i1i1_func,
            output_shape=(1,),
            name='loss_func')([
                autoencoder(img1_input),
                img1_input])
        model = Model(
            inputs=[img1_input],
            outputs=[loss_out],
            name=model_name)
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
    
    loss_out = Lambda(
        loss_func,
        output_shape=(1,),
        name='loss_func')(
        [autoencoder(inputs[0]), char_emb_model(inputs[0])])
    model = Model(inputs=inputs, output=[loss_out])
    
    optimizer = 'rmsprop'
    model.compile(
        optimizer=optimizer,
        loss={'loss_func' : lambda y_true, y_pred: y_pred})
    
    # ReduceLROnPlateau(patience=patience / 2, verbose=1),
    
    model_name = 'char-cnn'
    patience = 5
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto'),
        CSVLogger(filename=model_name + '.log.csv'),
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint(model_name + '.check',
                        save_best_only=True,
                        save_weights_only=True)]
    
    model.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        batch_size=batch_size,
        epochs=max_epochs,
        shuffle=True,
        callbacks=callbacks)







    num_matrices = 2 if '2' in args.loss_type else 1
    data_train = DataGeneratorPhraseImIm(
        img_dir,
        index_fname,
        batch_size=args.batch_size,
        output_filler=True,
        num_matrices=num_matrices)
    data_val = DataGeneratorPhraseImIm(
        img_dir,
        index_fname.replace('train', 'trial'),
        batch_size=args.batch_size_val,
        shuffle=False,
        output_filler=True,
        num_matrices=num_matrices)

    try:
        model.fit_generator(
            generator=data_train.generate(),
            validation_data=data_val.generate(),
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
            shuffle=True,
            callbacks=callback)
        print('Model wrote to {0}.check'.format(model_name))
    except:
        pass
    finally:
        data_train.save_log(model_basename + '_samples_train.json')
        data_val.save_log(model_basename + '_samples_val.json')
