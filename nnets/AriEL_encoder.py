import logging

import numpy as np
import tensorflow as tf
from prettytable import PrettyTable

from GenericTools.KerasTools.esoteric_layers.convenience_layers import predefined_model, ExpandDims, Slice
from GenericTools.KerasTools.convenience_operations import dynamic_fill, slice_, dynamic_zeros, dynamic_ones, \
    dynamic_filler, replace_column, slice_from_to

tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Layer, RNN

from DifferentiableAriEL.nnets.tf_tools.tf_helpers import tf_update_bounds_encoder
from DifferentiableAriEL.nnets.tf_tools.keras_layers import UpdateBoundsEncoder

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def DAriEL_Encoder_model(vocab_size=101,
                         emb_dim=2,
                         lat_dim=4,
                         language_model=None,
                         maxlen=6,
                         PAD=None):
    layer = ArielEncoderLayer0(vocab_size=vocab_size, emb_dim=emb_dim,
                                   lat_dim=lat_dim, language_model=language_model,
                                   maxlen=maxlen, PAD=PAD)
    input_questions = Input(shape=(None,), name='question')
    point = layer(input_questions)
    model = Model(inputs=input_questions, outputs=point)
    return model


# FIXME: encoder
class ArielEncoderLayer0(object):

    def __init__(self,
                 vocab_size=101,
                 emb_dim=2,
                 lat_dim=4,
                 language_model=None,
                 size_lat_dim=0,
                 maxlen=6,
                 PAD=None,
                 softmaxes=False):

        self.__dict__.update(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             language_model=language_model,
                             maxlen=maxlen,
                             PAD=PAD,
                             softmaxes=softmaxes)

        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)

        if self.PAD == None: logger.warn('Since the PAD was not specified we assigned a value of zero!'); self.PAD = 0

    def __call__(self, input_questions):

        start_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.PAD)})(input_questions)
        start_layer = Lambda(K.squeeze, arguments={'axis': 1})(start_layer)

        softmax = self.language_model(start_layer)

        expanded_os = ExpandDims(1)(softmax)
        final_softmaxes = expanded_os

        for final in range(self.maxlen):
            partial_question = Slice(1, 0, final + 1)(input_questions)
            softmax = self.language_model(partial_question)
            expanded_os = ExpandDims(1)(softmax)
            final_softmaxes = Concatenate(axis=1)([final_softmaxes, expanded_os])

        final_softmaxes = Lambda(slice_)(final_softmaxes)

        point = probsToPoint(self.vocab_size, self.lat_dim)([final_softmaxes, input_questions])

        if not self.softmaxes:
            return point
        else:
            return point, final_softmaxes


class probsToPoint(object):

    def __init__(self, vocab_size=2, lat_dim=3):
        # super(vAriEL_Encoder, self).__init__()
        self.__dict__.update(vocab_size=vocab_size, lat_dim=lat_dim)

    def __call__(self, inputs):
        softmax, input_questions = inputs

        # assert K.int_shape(softmax)[1] == K.int_shape(input_questions)[1]

        def downTheTree(inputs):
            listProbs, listTokens = inputs

            # for the matrix multiplications that follow we are not going to 
            # use the output of the LSTM after the last token has passed
            # listProbs = listProbs[:, :-1, :]

            cumsums = tf.cumsum(listProbs, axis=2, exclusive=True)
            # for p_ij, c_ij, token_i in zip(listProbs, cumsums, listTokens):

            listTokens = tf.cast(listTokens, dtype=tf.int32)
            one_hot = K.one_hot(listTokens, self.vocab_size)

            p_iti = K.sum(listProbs * one_hot, axis=2)
            c_iti = K.sum(cumsums * one_hot, axis=2)

            # Create another vector containing zeroes to pad `a` to (2 * 3) elements.
            zero_padding = Lambda(dynamic_zeros,
                                  arguments={'d': self.lat_dim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(p_iti)
            zero_padding = K.squeeze(zero_padding, axis=1)
            ones_padding = Lambda(dynamic_ones, arguments={'d': self.lat_dim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(
                p_iti)
            ones_padding = K.squeeze(ones_padding, axis=1)

            # Concatenate `a_as_vector` with the padding.
            p_padded = tf.concat([p_iti, ones_padding], 1)
            c_padded = tf.concat([c_iti, zero_padding], 1)

            # Reshape the padded vector to the desired shape.
            p_latent = tf.reshape(p_padded, [-1, tf.shape(p_iti)[1], self.lat_dim])
            c_latent = tf.reshape(c_padded, [-1, tf.shape(c_iti)[1], self.lat_dim])

            # calculate the final position determined by AriEL
            p_cumprod = tf.math.cumprod(p_latent, axis=1, exclusive=True)
            p_prod = tf.reduce_prod(p_latent, axis=1)
            cp = c_latent * p_cumprod

            lowBound = tf.reduce_sum(cp, axis=1)

            point = lowBound + p_prod / 2

            return point

        pointLatentDim = Lambda(downTheTree, name='downTheTree')([softmax, input_questions])
        return pointLatentDim


class ArielEncoderLayer1(object):
    """ simpler version of the encoder where I strictly do what the algorithm
    in the paper says """

    def __init__(self,
                 vocab_size=101,
                 emb_dim=2,
                 lat_dim=4,
                 maxlen=10,
                 language_model=None,
                 PAD=None,
                 size_lat_dim=3,
                 output_type='both'):

        self.__dict__.update(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             maxlen=maxlen,
                             language_model=language_model,
                             PAD=PAD,
                             size_lat_dim=size_lat_dim,
                             output_type=output_type)

        # if the input is a rnn, use that, otherwise use an LSTM

        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)

        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')

    def __call__(self, input_question):
        PAD_layer = Lambda(dynamic_filler, arguments={'d': 1, 'value': float(self.PAD)})(input_question)

        sentence_layer = Concatenate(axis=1)([PAD_layer, input_question])
        sentence_layer = Lambda(tf.cast, arguments={'dtype': tf.int32, })(sentence_layer)

        low_bound = Lambda(dynamic_filler, arguments={'d': self.lat_dim, 'value': 0.})(input_question)
        upp_bound = Lambda(dynamic_filler, arguments={'d': self.lat_dim, 'value': float(self.size_lat_dim)})(
            input_question)

        curDim = 0
        for j in range(self.maxlen - 1):
            s_0toj = Slice(1, 0, j + 1)(sentence_layer)
            s_j = Slice(1, j + 1, j + 2)(sentence_layer)
            softmax = self.language_model(s_0toj)
            low_bound, upp_bound = UpdateBoundsEncoder(self.lat_dim, self.vocab_size, curDim)(
                [low_bound, upp_bound, softmax, s_j])

            # NOTE: at each iteration, change the dimension
            curDim += 1
            if curDim >= self.lat_dim:
                curDim = 0

        z = tf.add(low_bound, upp_bound) / 2

        return z


def ArielEncoderLayer2(
        vocab_size=3,
        emb_dim=3,
        lat_dim=3,
        maxlen=3,
        size_lat_dim=1.,
        language_model=None,
        PAD=None):
    cell = ArielEncoderCell2(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             maxlen=maxlen,
                             size_lat_dim=size_lat_dim,
                             language_model=language_model,
                             PAD=PAD)
    rnn = RNN([cell], return_sequences=False, return_state=False, name='AriEL_encoder')

    input_question = Input(shape=(None,), name='question')
    expanded = ExpandDims(axis=2)(input_question)
    o_s = rnn(expanded)
    model = Model(inputs=input_question, outputs=o_s)

    return model


class ArielEncoderCell2(Layer):

    def __init__(self,
                 vocab_size=101,
                 emb_dim=2,
                 lat_dim=4,
                 maxlen=3,
                 language_model=None,
                 size_lat_dim=3,
                 PAD=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.__dict__.update(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             maxlen=maxlen,
                             language_model=language_model,
                             size_lat_dim=size_lat_dim,
                             PAD=PAD)

        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)

        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')

    def build(self, input_shape):
        super(ArielEncoderCell2, self).build(input_shape)  # Be sure to call this at the end

    @property
    def state_size(self):
        return (self.lat_dim,
                self.lat_dim,
                self.maxlen + 1,
                self.lat_dim,
                1,
                1)

    @property
    def output_size(self):
        return self.lat_dim

    def call(self, inputs, state):

        input_token = tf.squeeze(inputs, axis=1)
        low_bound, upp_bound, tokens, z, curDimVector, timeStepVector = state

        curDim = curDimVector[0]
        timeStep = timeStepVector[0]
        timeStep_plus1 = tf.squeeze(tf.cast(tf.add(timeStep, 1), dtype=tf.int32))
        timeStep_plus2 = tf.squeeze(tf.cast(tf.add(timeStep, 2), dtype=tf.int32))
        tf_curDim = tf.squeeze(tf.cast(curDim, dtype=tf.int32))

        tokens = replace_column(tokens, input_token, timeStep_plus1)

        initial_low_bound = dynamic_filler(batch_as=input_token, d=self.lat_dim, value=0.)
        initial_upp_bound = dynamic_filler(batch_as=input_token, d=self.lat_dim, value=float(self.size_lat_dim))

        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)
        low_bound = tf.cond(pred_t, lambda: low_bound, lambda: initial_low_bound, name='low_bound_cond')
        upp_bound = tf.cond(pred_t, lambda: upp_bound, lambda: initial_upp_bound, name='upp_bound_cond')

        s_0toj = slice_from_to(tokens, 0, timeStep_plus1)
        s_j = slice_from_to(tokens, timeStep_plus1, timeStep_plus2)
        s_j = tf.cast(s_j, dtype=tf.int32)

        s_0toj_layer = Input(tensor=s_0toj)
        softmax = self.language_model(s_0toj_layer)

        low_bound, upp_bound = tf_update_bounds_encoder(low_bound, upp_bound, softmax, s_j, tf_curDim)

        bounds = tf.concat([low_bound[..., tf.newaxis], upp_bound[..., tf.newaxis]], axis=2)
        z = tf.reduce_mean(bounds, axis=2)

        # NOTE: at each iteration, change the dimension, and add a timestep
        lat_dim = tf.cast(tf.shape(z)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(lat_dim)  # tf.math.greater_equal(curDim, lat_dim)
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)

        b = tf.shape(z)[0]
        curDimVector = tf.tile(curDim[tf.newaxis, :], [b, 1])
        timeStepVector = tf.tile(timeStep[tf.newaxis, :], [b, 1])

        new_state = [low_bound, upp_bound, tokens, z, curDimVector, timeStepVector]
        output = z

        return output, new_state


def test():
    vocab_size, batch_size, maxlen = 3, 6, 5
    lat_dim = 2

    input_questions = Input((None,))
    encoded = ArielEncoderLayer1(PAD=0,
                                 vocab_size=vocab_size,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen, )(input_questions)
    model = Model(input_questions, encoded)

    sentences = np.random.randint(vocab_size, size=(batch_size, maxlen))

    prediction = model.predict(sentences)

    t = PrettyTable()

    results = [sentences] [prediction]
    for a in zip(*results):
        t.add_row([y for y in a])

    t.add_row([y.shape for y in results])
    print(t)


if __name__ == '__main__':
    test()
