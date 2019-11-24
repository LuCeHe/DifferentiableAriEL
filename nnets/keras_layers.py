import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, \
                         Activation, Concatenate, Dense
from DifferentiableAriEL.nnets.tf_helpers import slice_from_to, clip_layer


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

