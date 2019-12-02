
import numpy as np
from numpy.random import seed
import logging
from tqdm import tqdm

import tensorflow as tf
from prettytable import PrettyTable

tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, RepeatVector, \
                         Activation, Concatenate, Dense, RNN, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework import function

from DifferentiableAriEL.nnets.tf_helpers import slice_, dynamic_ones, dynamic_one_hot, onehot_pseudoD, \
    pzToSymbol_withArgmax, clip_layer, dynamic_fill, dynamic_zeros, \
    pzToSymbolAndZ
from DifferentiableAriEL.nnets.keras_layers import ExpandDims, Slice

seed(3)
tf.set_random_seed(2)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class DAriEL_Decoder_Layer_2(Layer):

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 max_senLen = 3,
                 language_model=None,
                 PAD=None,
                 **kwargs):  
        super(DAriEL_Decoder_Layer_2, self).__init__(**kwargs)
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             max_senLen=max_senLen,
                             language_model=language_model,
                             PAD=PAD)
        
        # if the input is a rnn, use that, otherwise use an LSTM        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)          
        
        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')
        
    def build(self, input_shape):        
        super(DAriEL_Decoder_Layer_2, self).build(input_shape)  # Be sure to call this at the end
        
    @property
    def state_size(self):
        return (self.vocabSize,
                self.max_senLen,
                self.latDim,
                1,
                1)

    @property
    def output_size(self):
        return self.vocabSize
    
    def call(self, inputs, state):

        input_point = inputs
        one_softmax, tokens, unfolding_point, curDimVector, timeStepVector = state
        
        curDim = curDimVector[0]
        timeStep = timeStepVector[0]
        
        # initialization        
        PAD_layer = Input(tensor=self.PAD*tf.ones_like(input_point[:,0, tf.newaxis]))
        initial_softmax = self.language_model(PAD_layer)
        
        # FIXME: it would be interesting to consider what would happen if we feed different points within
        # a batch
        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)

        unfolding_point = tf.cond(pred_t, lambda: input_point, lambda: unfolding_point, name='unfolding_point')
        one_softmax = tf.cond(pred_t, lambda: initial_softmax, lambda: one_softmax, name='one_softmax')
        #tokens = tf.cond(pred_t, lambda: PAD_layer, lambda: tokens, name='tokens')

        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None, 1))
        token = tf.squeeze(token, axis=1)
        tokens = replace_column(tokens, token, timeStep)
        
        # get the softmax for the next iteration
        # make sure you feed only up to the tokens that have been produced now ([:timeStep]
        # otherwise you are feeding a sentence with tons of zeros at the end. 
        tokens_in = Input(tensor=tokens[:, :tf.cast(tf.squeeze(timeStep), dtype=tf.int64)+1])
        #tokens_in = Input(tensor=tokens[:, tf.cast(tf.squeeze(timeStep), dtype=tf.int64)+1, tf.newaxis])
        one_softmax = self.language_model(tokens_in)        
        
        # NOTE: at each iteration, change the dimension, and add a timestep
        latDim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(latDim)  # tf.math.greater_equal(curDim, latDim)    
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)
        
        b = tf.shape(one_softmax)[0]
        curDimVector = tf.tile(curDim[tf.newaxis,:], [b, 1])
        timeStepVector = tf.tile(timeStep[tf.newaxis,:], [b, 1])
        
        new_state = [one_softmax, tokens, unfolding_point, curDimVector, timeStepVector]
        output = one_softmax
        
        return output, new_state
          



def DAriEL_Decoder_model(vocabSize=101,
                         embDim=2,
                         latDim=4,
                         max_senLen=10,
                         language_model=None,
                         PAD=None,
                         output_type='both'):  
    
    layer = DAriEL_Decoder_Layer(vocabSize=vocabSize, embDim=embDim,
                                 latDim=latDim, max_senLen=max_senLen,
                                 language_model=language_model, PAD=PAD,
                                 output_type=output_type)
    input_point = Input(shape=(latDim,), name='input_point')
    discrete_sequence_output = layer(input_point)    
    model = Model(inputs=input_point, outputs=discrete_sequence_output)
    return model


class DAriEL_Decoder_Layer(object):

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 max_senLen=10,
                 language_model=None,
                 PAD=None,
                 output_type='both'):  
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             max_senLen=max_senLen,
                             language_model=language_model,
                             PAD=PAD,
                             output_type=output_type)
        
        # if the input is a rnn, use that, otherwise use an LSTM
        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)          
        
        if self.PAD == None: raise ValueError('Define the startId you are using ;) ')
    
    def __call__(self, input_point):
            
        # FIXME: I think arguments passed this way won't be saved with the model
        # follow instead: https://github.com/keras-team/keras/issues/1879
        # RNN_starter = Lambda(dynamic_zeros, arguments={'d': self.embDim})(input_point)   
        # RNN_starter = Lambda(dynamic_fill, arguments={'d': self.embDim, 'value': .5})(input_point)

        output = pointToProbs(vocabSize=self.vocabSize,
                              latDim=self.latDim,
                              embDim=self.embDim,
                              max_senLen=self.max_senLen,
                              language_model=self.language_model,
                              PAD=self.PAD,
                              output_type=self.output_type)(input_point)
    
        return output


class pointToProbs(object):

    def __init__(self,
                 vocabSize=2,
                 latDim=3,
                 embDim=2,
                 max_senLen=10,
                 language_model=None,
                 PAD=None,
                 output_type='both'):
        """        
        inputs:
            output_type: 'tokens', 'softmaxes' or 'both'
        """
        self.__dict__.update(vocabSize=vocabSize, latDim=latDim,
                             embDim=embDim, max_senLen=max_senLen,
                             language_model=language_model,
                             PAD=PAD,
                             output_type=output_type)
    
    def __call__(self, inputs):
        input_point = inputs
        
        start_layer = Lambda(dynamic_fill, arguments={'d': self.max_senLen, 'value': float(self.PAD)})(input_point)
        start_layer = Lambda(K.squeeze, arguments={'axis': 1})(start_layer)
        # startId_layer = Lambda(tf.cast, arguments={'dtype': tf.int64})(startId_layer)
        
        initial_softmax = self.language_model(start_layer)    
        one_softmax = initial_softmax
        
        # by clipping the values, it can accept inputs that go beyong the 
        # unit hypercube
        clipped_layer = Lambda(clip_layer, arguments={'min_value': 0., 'max_value': 1.})(inputs)  # Clip(0., 1.)(input_point)
        
        unfolding_point = clipped_layer
        
        expanded_os = ExpandDims(1)(one_softmax)
        final_softmaxes = expanded_os
        final_tokens = None  # startId_layer
        curDim = 0
        curDim_t = tf.constant(curDim)
        
        # NOTE: since ending on the EOS token would fail for mini-batches, 
        # the algorithm stops at a maxLen when the length of the sentence 
        # is maxLen
        for _ in range(self.max_senLen):                
            
            token, unfolding_point = Lambda(pzToSymbolAndZ)([one_softmax, unfolding_point, curDim_t])
            token.set_shape((None, 1))
            # output = Lambda(create_new_token)([one_softmax, unfolding_point])
            
            if final_tokens == None:
                final_tokens = token
            else:
                final_tokens = Concatenate(axis=1)([final_tokens, token])
            
            # FIXME: optimal way would be to cut the following tensor in order to be         
            # of length max_senLen    
            tokens = Concatenate(axis=1)([final_tokens, start_layer])
            
            # get the softmax for the next iteration
            one_softmax = self.language_model(tokens)

            expanded_os = ExpandDims(1)(one_softmax)
            final_softmaxes = Concatenate(axis=1)([final_softmaxes, expanded_os])
            
            # NOTE: at each iteration, change the dimension
            curDim += 1
            if curDim >= self.latDim:
                curDim = 0
            
            curDim_t = tf.constant(curDim)
        
        # remove last softmax, since the initial was given by the an initial
        # zero vector
        softmaxes = Lambda(slice_)(final_softmaxes)
        # softmaxes = final_softmaxes
        tokens = final_tokens

        # FIXME: give two options: the model giving back the whole softmaxes
        # sequence, or the model giving back the sequence of tokens 
        
        if self.output_type == 'tokens':
            output = tokens
        elif self.output_type == 'softmaxes':
            output = softmaxes
        elif self.output_type == 'both':
            output = [tokens, softmaxes]
        else:
            raise ValueError('the output_type specified is not implemented!')
        
        return output

