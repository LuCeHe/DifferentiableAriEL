
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

from DifferentiableAriEL.nnets.tf_tools.tf_helpers import slice_, dynamic_ones, dynamic_one_hot, onehot_pseudoD, \
    pzToSymbol_withArgmax, clip_layer, dynamic_fill, dynamic_filler, dynamic_zeros, \
    pzToSymbolAndZ
from DifferentiableAriEL.nnets.tf_tools.keras_layers import ExpandDims, Slice, predefined_model, UpdateBoundsEncoder

seed(3)
tf.set_random_seed(2)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def DAriEL_Encoder_model(vocabSize=101,
                         embDim=2,
                         latDim=4,
                         language_model=None,
                         max_senLen=6,
                         PAD=None):      
    
    layer = DAriEL_Encoder_Layer_0(vocabSize=vocabSize, embDim=embDim,
                                   latDim=latDim, language_model=language_model,
                                   max_senLen=max_senLen, PAD=PAD)        
    input_questions = Input(shape=(None,), name='question')    
    point = layer(input_questions)
    model = Model(inputs=input_questions, outputs=point)
    return model


# FIXME: encoder
class DAriEL_Encoder_Layer_0(object):

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 language_model=None,
                 max_senLen=6,
                 PAD=None,
                 softmaxes=False):

        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             language_model=language_model,
                             max_senLen=max_senLen,
                             PAD=PAD,
                             softmaxes=softmaxes)
        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)           
            
        if self.PAD == None: logger.warn('Since the PAD was not specified we assigned a value of zero!'); self.PAD = 0
        
    def __call__(self, input_questions):
                
        start_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.PAD)})(input_questions)
        start_layer = Lambda(K.squeeze, arguments={'axis': 1})(start_layer)
            
        softmax = self.language_model(start_layer)
        
        expanded_os = ExpandDims(1)(softmax)
        final_softmaxes = expanded_os
        
        for final in range(self.max_senLen):  
            partial_question = Slice(1, 0, final + 1)(input_questions)
            softmax = self.language_model(partial_question)
            expanded_os = ExpandDims(1)(softmax)
            final_softmaxes = Concatenate(axis=1)([final_softmaxes, expanded_os])
            
        final_softmaxes = Lambda(slice_)(final_softmaxes)
        
        point = probsToPoint(self.vocabSize, self.latDim)([final_softmaxes, input_questions])

        if not self.softmaxes:
            return point
        else:
            return point, final_softmaxes


class probsToPoint(object):

    def __init__(self, vocabSize=2, latDim=3):
        # super(vAriEL_Encoder, self).__init__()
        self.__dict__.update(vocabSize=vocabSize, latDim=latDim)
    
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
            one_hot = K.one_hot(listTokens, self.vocabSize)
            
            p_iti = K.sum(listProbs * one_hot, axis=2)
            c_iti = K.sum(cumsums * one_hot, axis=2)
            
            # Create another vector containing zeroes to pad `a` to (2 * 3) elements.
            zero_padding = Lambda(dynamic_zeros, arguments={'d': self.latDim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(p_iti)
            zero_padding = K.squeeze(zero_padding, axis=1)
            ones_padding = Lambda(dynamic_ones, arguments={'d': self.latDim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(p_iti)
            ones_padding = K.squeeze(ones_padding, axis=1)
            
            # Concatenate `a_as_vector` with the padding.
            p_padded = tf.concat([p_iti, ones_padding], 1)
            c_padded = tf.concat([c_iti, zero_padding], 1)
            
            # Reshape the padded vector to the desired shape.
            p_latent = tf.reshape(p_padded, [-1, tf.shape(p_iti)[1], self.latDim])
            c_latent = tf.reshape(c_padded, [-1, tf.shape(c_iti)[1], self.latDim])
            
            # calculate the final position determined by AriEL
            p_cumprod = tf.math.cumprod(p_latent, axis=1, exclusive=True)
            p_prod = tf.reduce_prod(p_latent, axis=1)
            cp = c_latent * p_cumprod
            
            lowBound = tf.reduce_sum(cp, axis=1)
            
            point = lowBound + p_prod / 2

            return point
                
        pointLatentDim = Lambda(downTheTree, name='downTheTree')([softmax, input_questions])
        return pointLatentDim


class DAriEL_Encoder_Layer_1(object):
    """ simpler version of the encoder where I strictly do what the algorithm
    in the paper says """

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 max_senLen=10,
                 language_model=None,
                 PAD=None,
                 size_latDim=3,
                 output_type='both'):
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             max_senLen=max_senLen,
                             language_model=language_model,
                             PAD=PAD,
                             size_latDim=size_latDim,
                             output_type=output_type)
        
        # if the input is a rnn, use that, otherwise use an LSTM
        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)          
        
        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')
    
    def __call__(self, input_question):
        PAD_layer = Lambda(dynamic_filler, arguments={'d': 1, 'value': float(self.PAD)})(input_question)
        
        sentence_layer = Concatenate(axis=1)([PAD_layer, input_question])
        sentence_layer = Lambda(tf.cast, arguments={'dtype': tf.int32, })(sentence_layer)
        
        low_bound = Lambda(dynamic_filler, arguments={'d': self.latDim, 'value': 0.})(input_question)
        upp_bound = Lambda(dynamic_filler, arguments={'d': self.latDim, 'value': float(self.size_latDim)})(input_question)
        
        curDim = 0
        for j in range(self.max_senLen - 1):  
            s_0toj = Slice(1, 0, j + 1)(sentence_layer)
            s_j = Slice(1, j + 1, j + 2)(sentence_layer)
            softmax = self.language_model(s_0toj)
            low_bound, upp_bound = UpdateBoundsEncoder(self.latDim, self.vocabSize, curDim)([low_bound, upp_bound, softmax, s_j])

            # NOTE: at each iteration, change the dimension
            curDim += 1
            if curDim >= self.latDim:
                curDim = 0

        z = tf.add(low_bound, upp_bound) / 2
        
        return z


def test():
    
    vocabSize, batch_size, max_senLen = 3, 6, 5
    latDim = 2
                     
    input_questions = Input((None,))
    encoded = DAriEL_Encoder_Layer_1(PAD=0,
                                     vocabSize=vocabSize,
                                     latDim=latDim,
                                     max_senLen=max_senLen,)(input_questions)
    model = Model(input_questions, encoded)
    
    sentences = np.random.randint(vocabSize, size=(batch_size, max_senLen))

    prediction = model.predict(sentences)
    
    t = PrettyTable()
    
    results = [sentences] + [prediction]
    for a in zip(*results):
        t.add_row([y for y in a])
    
    t.add_row([y.shape for y in results])
    print(t)
    
    
if __name__ == '__main__':
    test()
