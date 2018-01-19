
from keras import backend as K
from keras.layers import Embedding, Flatten, UpSampling1D, Reshape
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Subtract
from keras.layers import Dot
from keras.models import Model
from keras.regularizers import l2

from preprocessing import char_indices

def make_siamese_model(input1, input2, encoder, l2_strength=1e-3, dense_dim=128):
    in1 = encoder(input1)
    in2 = encoder(input2)
    in1 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_src')(in1)
    in2 = Dense(
        dense_dim,
        kernel_regularizer=l2(l2_strength),
        bias_regularizer=l2(l2_strength),
        activation='relu',
        name='siam_dense1_trg')(in2)
    simil = Dot(
        axes=(1,1),
        normalize=True,
        name='siam_cos_simil')([in1, in2])
    siamese = Model(
        inputs=[in1_input, in2_input],
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
    In this loss function, we apply a different Dense layer
    to each encoded image before computing its cosine distance.
    This is an attempt to correct the fact that the smallest
    cosine distance of two encoded (different) images is when
    the encoder produces zero values. However, the parameter
    values that cause such situation destroy the information
    necessary to reconstruct the image in the auto-encoder regime,
    which is a conflicting loss factor.
    """
    auto_img1, img1, cosine_simil = args
    auto_img1 = Flatten()(auto_img1)
    img1 = Flatten()(img1)
    auto_loss = K.mean(K.square(auto_img1 - img1), axis=-1)
    cosine_simil = K.mean(cosine_simil, axis=-1)
    loss = Subtract()([Reshape((1,))(auto_loss), Reshape((1,))(cosine_simil)])
    print('Loss shape: {0}'.format(loss.shape))
    return loss

def make_encoder(
    maxlen,
    char_emb_size,
    num_filters=(64, 100),
    filter_lengths=(3, 3),
    subsamples=(2, 1),
    pool_lengths=(2, 2)):
    """
    Make the output of an encoder (but does not build the model).
    """
    entity = Input(shape=(maxlen,), dtype='int32')

    char_emb = Embedding(
        input_dim=len(char_indices),
        output_dim=char_emb_size,
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
    x = Dense(128, activation='relu', name='enc_dense_128')(x)

    inputs = [entity]
    outputs = [x]
    return inputs, outputs, char_emb_x

def make_decoder(
    encoded_entity,
    num_filters=(64, 100),
    filter_lengths=(3, 3),
    subsamples=(2, 1),
    up_lengths=(2, 2)):

    x = Dense(2 * num_filters[0], activation='relu', name='dec_dense')(encoded_entity)
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

