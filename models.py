
import numpy as np

from keras import backend as K
from keras.layers import Embedding, Flatten, UpSampling1D, Reshape
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import Lambda, concatenate
from keras.layers import TimeDistributed
from keras.layers import Add
from keras.layers import Subtract
from keras.layers import Dot
from keras.losses import mean_squared_error
from keras.models import Model
from keras.regularizers import l2

from preprocessing import char_indices

L2Strength = 1e-6

def make_label_input(maxlen):
    label_input = Input(shape=(maxlen,), dtype='int32')
    return label_input

def make_siamese_model2(input1, input2, encoder, l2_strength=1e-6, dense_dim=128, drop=0.3):
    """
    Siamese model that computes cosine *similarity* (not distance).
    """
    in1 = encoder(input1)
    in2 = encoder(input2)
    in1 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_src')(in1)
    in1 = Dropout(drop, name='siam_drop1_src')(in1)
    in2 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_trg')(in2)
    in2 = Dropout(drop, name='siam_drop2_src')(in2)
    return in1, in2

def make_siamese_model(input1, input2, encoder, l2_strength=1e-6, dense_dim=128, drop=0.3):
    """
    Siamese model that computes cosine *similarity* (not distance).
    """
    in1 = encoder(input1)
    in2 = encoder(input2)
    in1 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_src')(in1)
    in1 = Dropout(drop, name='siam_drop1_src')(in1)
    in2 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_trg')(in2)
    in2 = Dropout(drop, name='siam_drop2_src')(in2)
    simil = Dot(
        axes=(1,1),
        normalize=True,
        name='siam_cos_simil')([in1, in2])
    siamese = Model(
        inputs=[input1, input2],
        outputs=[simil],
        name='siamese_simil')
    return siamese

def loss_i1i1_func(args):
    decoded_label, embedded_label = args
    decoded_label = Flatten(name='loss_dec_flatten')(decoded_label)
    embedded_label = Flatten(name='loss_emb_flatten')(embedded_label)
    loss = mean_squared_error(decoded_label, embedded_label)
    print('Loss shape: {0}'.format(loss.shape))
    return loss

def loss_i1i1_i1i2s_func(args):
    """
    In this loss function, we use the cosine similarity (in negative form)
    as part of the loss.
    """
    auto_in1, in1, cosine_simil = args
    auto_in1 = Flatten()(auto_in1)
    in1 = Flatten()(in1)
    auto_loss = K.mean(K.square(auto_in1 - in1), axis=-1)
    cosine_simil = K.mean(cosine_simil, axis=-1)
    loss = Subtract()([Reshape((1,))(auto_loss), Reshape((1,))(cosine_simil)])
    print('Loss shape: {0}'.format(loss.shape))
    return loss

def loss_i1i1_i1i2s_i1j1s_func(args):
    """
    In this loss function, we use the cosine similarity (in negative form)
    as part of the loss and the cosine similarity of negative samples
    (in positive form) also as part of the loss. This is inspired by the
    Noise Contrastive Estimation.
    """
    auto_in1, in1, cosine_simil_pos, cosine_simil_neg = args
    auto_in1 = Flatten()(auto_in1)
    in1 = Flatten()(in1)
    auto_loss = K.mean(K.square(auto_in1 - in1), axis=-1)
    cosine_simil_pos = K.mean(cosine_simil_pos, axis=-1)
    cosine_simil_neg = K.mean(cosine_simil_neg, axis=-1)
    loss = Subtract()([Reshape((1,))(auto_loss), Reshape((1,))(cosine_simil_pos)])
    loss = Add()([loss, Reshape((1,))(cosine_simil_neg)])
    print('Loss shape: {0}'.format(loss.shape))
    return loss

def make_encoder(
    maxlen=16,
    char_emb_size=128,
    num_filters=(128, 128 * 2, 128 * 4),
    filter_lengths=(3, 3, 3),
    subsamples=(1, 1, 1),
    pool_lengths=(2, 2, 2)):
    """
    Make the output of an encoder (but does not build the model).
    """
    entity = make_label_input(maxlen)

    # TODO: make explicit 0 embedding for character index 0.
    embs = np.random.uniform(-0.05, 0.05, (len(char_indices) + 1, char_emb_size))
    embs[0, :] = 0
    char_emb = Embedding(
        input_dim=len(char_indices) + 1,
        output_dim=char_emb_size,
        weights=[embs],
        trainable=False,
        name='char_emb')
    char_emb_x = char_emb(entity)
    x = char_emb_x

    for i in range(len(num_filters)):
        x = Conv1D(
            filters=num_filters[i],
            kernel_size=filter_lengths[i],
            padding='same',
            activation='relu',
            strides=subsamples[i],
            name='enc_conv1d_' + str(i))(x)
        x = MaxPooling1D(
            pool_size=pool_lengths[i],
            name='enc_maxpool1d_' + str(i))(x)
    x = Flatten(name='enc_flatten')(x)
    x = Dense(
            128,
            kernel_regularizer=l2(L2Strength),
            bias_regularizer=l2(L2Strength),
            activation='relu',
            name='enc_dense_128')(x)

    inputs = [entity]
    outputs = [x]
    return inputs, outputs, char_emb_x

def make_decoder(
    encoded_entity,
    num_filters=(64, 100),
    filter_lengths=(3, 3),
    subsamples=(2, 1),
    up_lengths=(2, 2)):

    x = Dense(
        2 * num_filters[0],
        kernel_regularizer=l2(L2Strength),
        bias_regularizer=l2(L2Strength),
        activation='tanh',
        name='dec_dense')(encoded_entity)
    x = Reshape((2, num_filters[0]), name='dec_reshape')(x)

    for i in range(len(num_filters)):
        x = Conv1D(
            filters=num_filters[i],
            kernel_size=filter_lengths[i],
            padding='same',
            activation='relu',
            strides=subsamples[i],
            name='dec_conv1d_' + str(i))(x)
        x = UpSampling1D(
            size=up_lengths[i],
            name='dec_up1d_' + str(i))(x)

    inputs = [encoded_entity]
    outputs = [x]
    return inputs, outputs

