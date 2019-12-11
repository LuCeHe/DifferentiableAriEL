import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, \
                         Activation, Concatenate, Dense
from tensorflow.python.keras.engine.base_layer import Layer

from DifferentiableAriEL.nnets.tf_tools.tf_helpers import slice_from_to, \
    clip_layer, pzToSymbol_withArgmax, replace_column

class ExpandDims(object):

    def __init__(self, axis):
        self.axis = axis
        
    def __call__(self, inputs):

        def ed(tensor, axis):
            expanded = K.expand_dims(tensor, axis=axis)
            return expanded
        
        return Lambda(ed, arguments={'axis': self.axis})(inputs)


class Slice(object):

    # axis parameter is not functional
    def __init__(self, axis, initial, final):
        self.axis, self.initial, self.final = axis, initial, final
        
    def __call__(self, inputs):
        return Lambda(slice_from_to, arguments={'initial': self.initial, 'final': self.final})(inputs)


class Clip(object):

    def __init__(self, min_value=0., max_value=1.):
        self.min_value, self.max_value = min_value, max_value
        
    def __call_(self, inputs):
        return Lambda(clip_layer, arguments={'min_value': self.min_value, 'max_value': self.max_value})(inputs)


"""
class TestActiveGaussianNoise(Layer):
    @interfaces.legacy_gaussiannoise_support
    def __init__(self, stddev, **kwargs):
        super(TestActiveGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(TestActiveGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
        
    
class SelfAdjustingGaussianNoise(Layer):
    @interfaces.legacy_gaussiannoise_support
    def __init__(self, tensor_type='scalar', **kwargs):
        super(SelfAdjustingGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        
        if not tensor_type in ['scalar', 'tensor']: 
            raise ValueError("tensor_type can be either 'scalar' or 'tensor'!")
            
        self.tensor_type = tensor_type
        
        
        self.stddev_initializer = keras.initializers.get('ones')
        self.stddev_regularizer = keras.regularizers.get(None)
        self.stddev_constraint = keras.constraints.get(None)


        
    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.tensor_type == 'scalar':
            stddev_value = tf.Variable([1.], dtype=tf.float32)
            self.stddev = tf.ones(shape,
                                  dtype=tf.float32)
            self.stddev *= stddev_value
            self.trainable_weights = [stddev_value]
            
        else:
            self.stddev = self.add_weight(shape=shape,
                                         initializer=self.stddev_initializer,
                                         regularizer=self.stddev_regularizer,
                                         constraint=self.stddev_constraint,
                                         name='gamma',
                                         )
        super(SelfAdjustingGaussianNoise, self).build(input_shape)

    def call(self, inputs, training=None):
        def noised():
            return inputs + self.stddev*K.random_normal(shape=K.shape(inputs),
                                                        mean=0.,
                                                        stddev=1.)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SelfAdjustingGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

"""


def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(256, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocabSize, activation='softmax')(lstm_output)
    
    return Model(inputs=input_question, outputs=softmax)


class UpdateBoundsEncoder(Layer):

    def __init__(self, latDim, vocabSize, curDim, **kwargs):
        super(UpdateBoundsEncoder, self).__init__(**kwargs)
        
        self.latDim, self.vocabSize, self.curDim = latDim, vocabSize, curDim

    def call(self, inputs, training=None):
        low_bound, upp_bound, softmax, s_j = inputs
        s = s_j[:, 0]
        d_oh = tf.one_hot(self.curDim * tf.ones_like(s), self.latDim)
        _d_oh = tf.subtract(tf.ones(self.latDim), d_oh, name='d_inv_oh')
        
        c_upp = K.cumsum(softmax, axis=1)
        c_low = tf.cumsum(softmax, axis=1, exclusive=True)
        range_ = upp_bound[:, self.curDim] - low_bound[:, self.curDim]
        
        s_oh = tf.one_hot(s, self.vocabSize)
        
        # tf convoluted way to assign a value to a location ,
        # to minimize time, I'll go to the first and fast solution
        
        # up bound
        upp_update = range_ * tf.reduce_sum(c_upp * s_oh, axis=1)
        updated_upp = tf.add(low_bound[:, self.curDim], upp_update)[:, tf.newaxis] * d_oh
        
        upp_bound = tf.add(upp_bound * _d_oh, updated_upp)
        
        # low bound        
        low_update = range_ * tf.reduce_sum(c_low * s_oh, axis=1)
        updated_low = tf.add(low_bound[:, self.curDim], low_update)[:, tf.newaxis] * d_oh
        
        low_bound = tf.add(low_bound * _d_oh, updated_low)

        # FIXME: final output is upp_bound, low_bound
        return [low_bound, upp_bound]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


class UpdateBoundsDecoder(Layer):

    def __init__(self, curDim, **kwargs):
        super(UpdateBoundsDecoder, self).__init__(**kwargs)
        
        self.curDim = curDim

    def call(self, inputs, training=None):
        low_bound, upp_bound, softmax = inputs
        
        c_upp = K.cumsum(softmax, axis=1)
        c_low = tf.cumsum(softmax, axis=1, exclusive=True)
        range_ = upp_bound[:, self.curDim] - low_bound[:, self.curDim]
        
        # tf convoluted way to assign a value to a location ,
        # to minimize time, I'll go to the first and fast solution
        
        # up bound
        upp_update = range_[:, tf.newaxis] * c_upp
        updated_upp = tf.add(low_bound[:, self.curDim, tf.newaxis], upp_update)
        
        # low bound        
        low_update = range_[:, tf.newaxis] * c_low
        updated_low = tf.add(low_bound[:, self.curDim, tf.newaxis], low_update)
        
        # FIXME: final output is upp_bound, low_bound
        return [updated_low, updated_upp]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


class FindSymbolAndBounds(Layer):

    def __init__(self, vocabSize, curDim, **kwargs):
        super(FindSymbolAndBounds, self).__init__(**kwargs)
        
        self.vocabSize, self.curDim = vocabSize, curDim

    def call(self, inputs, training=None):
        Ls, Us, low_bound, upp_bound, input_point = inputs
        
        s = pzToSymbol_withArgmax(Us, Ls, input_point[:, self.curDim, tf.newaxis])
        # s = tf.cast(s, dtype=tf.int32)
        s_oh = tf.one_hot(s, self.vocabSize)
        
        new_L_column = tf.reduce_sum(Ls * s_oh, axis=1)
        low_bound = replace_column(low_bound, new_L_column, self.curDim)

        new_U_column = tf.reduce_sum(Us * s_oh, axis=1)
        upp_bound = replace_column(upp_bound, new_U_column, self.curDim)

        return [s, low_bound, upp_bound]


class ReplaceColumn(Layer):

    def __init__(self, column_position, **kwargs):
        super(ReplaceColumn, self).__init__(**kwargs)
        
        self.column_position = column_position

    def call(self, inputs, training=None):
        matrix, column = inputs
        
        matrix = tf.cast(matrix, dtype=tf.float32)
        column = tf.cast(column, dtype=tf.float32)
        new_matrix = replace_column(matrix, column, self.column_position)
        new_matrix = tf.cast(new_matrix, dtype=tf.int32)
        return new_matrix

