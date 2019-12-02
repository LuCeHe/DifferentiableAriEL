import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, \
                         Activation, Concatenate, Dense
from DifferentiableAriEL.nnets.tf_tools.tf_helpers import slice_from_to, clip_layer


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


def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(256, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocabSize, activation='softmax')(lstm_output)
    
    return Model(inputs=input_question, outputs=softmax)

"""