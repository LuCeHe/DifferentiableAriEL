#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:18:53 2019

vAriEL for New Word Acquisition

0. if you want to put it on keras you need to use numbers and not words
     - maybe create a module inside vAriEL that transforms a sentences 
     generator into a numbers generator
1. [DONE] character level
2. [DONE] lose complete connection to grammar
3. [DONE] every node the same number of tokens
4. start with a number of tokens larger than necessary, and 
     - assign tokens to characters them upon visit, first come first served
5. probably I need a <START> and an <END> tokens

"""

import numpy as np
import logging

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, \
                         Activation, Concatenate, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework import function

from numpy.random import seed
from numba.testing.ddt import feed_data
seed(3)
from tensorflow import set_random_seed
set_random_seed(2)

logger = logging.getLogger(__name__)


def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(128, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocabSize, activation='softmax')(lstm_output)
    
    return Model(inputs=input_question, outputs=softmax)
                 

# FIXME: don't pass arguments as 
# Lambda(dynamic_zeros, arguments={'d': dimension})(input)
# since it might not be saved with the model
def dynamic_zeros(x, d):
    batch_size = tf.shape(x)[0]
    return tf.zeros(tf.stack([batch_size, 1, d]))


def dynamic_ones(x, d):
    batch_size = tf.shape(x)[0]
    return tf.ones(tf.stack([batch_size, 1, d]))


def dynamic_fill(x, d, value):
    batch_size = tf.shape(x)[0]
    return tf.fill(tf.stack([batch_size, 1, d]), value)


def dynamic_one_hot(x, d, pos):
    batch_size = tf.shape(x)[0]
    one_hots = tf.ones(tf.stack([batch_size, 1, d])) * tf.one_hot(pos, d)
    return one_hots


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


def slice_(x):
    return x[:, :-1, :]


def slice_from_to(x, initial, final):
    # None can be used where initial or final, so
    # [1:] = [1:None]
    return x[:, initial:final]


class Clip(object):
    def __init__(self, min_value=0., max_value=1.):
        self.min_value, self.max_value = min_value, max_value
        
    def __call_(self, inputs):
        return Lambda(clip_layer, arguments={'min_value': self.min_value, 'max_value': self.max_value})(inputs)


def clip_layer(inputs, min_value, max_value):            
    eps = .5e-6
    clipped_point = K.clip(inputs, min_value + eps, max_value - eps)
    return clipped_point

        


def DAriEL_Encoder_model(vocabSize=101,
                         embDim=2,
                         latDim=4,
                         language_model=None,
                         max_senLen=6,
                         startId=None):      
    
    layer = DAriEL_Encoder_Layer(vocabSize=vocabSize, embDim=embDim,
                                 latDim=latDim, language_model=language_model,
                                 max_senLen=max_senLen, startId=startId)        
    input_questions = Input(shape=(None,), name='question')    
    point = layer(input_questions)
    model = Model(inputs=input_questions, outputs=point)
    return model


# FIXME: encoder
class DAriEL_Encoder_Layer(object):

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 language_model=None,
                 max_senLen=6,
                 startId=None,
                 softmaxes=False):  

        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             language_model=language_model,
                             max_senLen=max_senLen,
                             startId=startId,
                             softmaxes=softmaxes)
        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)           
            
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
        if not self.startId == 0: raise ValueError('Currently the model works only for startId == 0 ;) ')
        
    def __call__(self, input_questions):
                
        startId_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.startId)})(input_questions)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)
            
        softmax = self.language_model(startId_layer)
        
        expanded_os = ExpandDims(1)(softmax)
        final_softmaxes = expanded_os
        
        question_segments = []
        for final in range(self.max_senLen):  
            partial_question = Slice(1, 0, final + 1)(input_questions)   
            question_segments.append(partial_question)
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
            
            listTokens = tf.to_int32(listTokens)
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
            p_cumprod = tf.cumprod(p_latent, axis=1, exclusive=True)
            p_prod = tf.reduce_prod(p_latent, axis=1)
            cp = c_latent * p_cumprod
            
            lowBound = tf.reduce_sum(cp, axis=1)
            
            point = lowBound + p_prod / 2

            return point
                
        pointLatentDim = Lambda(downTheTree, name='downTheTree')([softmax, input_questions])
        return pointLatentDim


def DAriEL_Decoder_model(vocabSize=101,
                         embDim=2,
                         latDim=4,
                         max_senLen=10,
                         language_model=None,
                         startId=None,
                         output_type='both'):  
    
    layer = DAriEL_Decoder_Layer(vocabSize=vocabSize, embDim=embDim,
                                 latDim=latDim, max_senLen=max_senLen,
                                 language_model=language_model, startId=startId,
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
                 startId=None,
                 output_type='both'):  
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             max_senLen=max_senLen,
                             language_model=language_model,
                             startId=startId,
                             output_type=output_type)
        
        # if the input is a rnn, use that, otherwise use an LSTM
        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)          
        
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
    
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
                              startId=self.startId,
                              output_type=self.output_type)(input_point)
    
        return output

# this method seems to be quite unstable given the division by probabilities
def pzToSymbol_noArgmax(cumsum, cumsum_exclusive, value_of_interest):
    # determine the token selected (2 steps: xor and token)
    # differentiable xor (instead of tf.logical_xor)
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    abs_sx = tf.abs(signed_xor)
    eps = 1e-5; abs_sx = K.clip(abs_sx, 0 + eps, 1e10 - eps)  #hack
    almost_xor = tf.divide(signed_xor, abs_sx)
    almost_xor = tf.add(almost_xor, -1)
    almost_xor = tf.divide(almost_xor, -2)
    oh_symbol = tf.abs(almost_xor)
    
    # differentiable argmax (instead of tf.argmax)    
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    abs_c_minus_v = tf.abs(c_minus_v)           
    eps = 1e-5; abs_c_minus_v = K.clip(abs_c_minus_v, 0 + eps, 1e10 - eps)  #hack
    almost_symbol = tf.divide(c_minus_v, abs_c_minus_v)
    almost_symbol = tf.divide(tf.add(almost_symbol, -1), -2)
    almost_symbol = tf.abs(almost_symbol)
    symbol = tf.reduce_sum(almost_symbol, axis=1)
    symbol = tf.expand_dims(symbol, axis=1)

    return symbol, oh_symbol

@function.Defun()
def argmaxPseudoGrad(cumsum, cumsum_exclusive, value_of_interest, grad):
    dE_dz = tf.cast(grad, dtype=tf.float32)
    #dE_dz = tf.expand_dims(dE_dz, axis=1)

    #c_minus_v = tf.subtract(cumsum, value_of_interest)
    #ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    #signed_xor = c_minus_v * ce_minus_c
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    dz_dc_scaled = tf.maximum(1 - signed_xor, 0)   # val_loss: 0.1689
    dz_dc_scaled = - 10*signed_xor   # worse than when noArgmax

    cumsum_grad = dE_dz * dz_dc_scaled #tf.zeros_like(cumsum_exclusive) #dE_dz * c_minus_v # * tf.ones_like(cumsum_exclusive)
    cumsum_exclusive_grad = tf.zeros_like(cumsum_exclusive) #dE_dz * ce_minus_c #tf.zeros_like(cumsum_exclusive)
    value_grad = tf.ones_like(value_of_interest) #dE_dz*tf.ones_like(value_of_interest)   # ones val_loss: 0.1689 | dE_dz*tf.ones_like(value_of_interest) not very good
    
    return [cumsum_grad, 
            cumsum_exclusive_grad,
            value_grad]

# this method seems to be quite unstable given the division by probabilities
@function.Defun(grad_func=argmaxPseudoGrad)
def pzToSymbol_withArgmax(cumsum, cumsum_exclusive, value_of_interest):
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    symbol = tf.argmin(signed_xor, axis=1)

    symbol = tf.expand_dims(symbol, axis=1)
    symbol = tf.cast(symbol, dtype=tf.float32)
    return symbol


def pzToSymbol_derivableMock(cumsum, cumsum_exclusive, value_of_interest):
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    symbol = tf.reduce_sum(signed_xor, axis=1)
    
    return [symbol, cumsum]

class pointToProbs(object):

    def __init__(self,
                 vocabSize=2,
                 latDim=3,
                 embDim=2,
                 max_senLen=10,
                 language_model=None,
                 startId=None,
                 output_type='both'):
        """        
        inputs:
            output_type: 'tokens', 'softmaxes' or 'both'
        """
        self.__dict__.update(vocabSize=vocabSize, latDim=latDim,
                             embDim=embDim, max_senLen=max_senLen,
                             language_model=language_model,
                             startId=startId,
                             output_type=output_type)
    
    def __call__(self, inputs):
        input_point = inputs
        
        startId_layer = Lambda(dynamic_fill, arguments={'d': self.max_senLen, 'value': float(self.startId)})(input_point)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)
        #startId_layer = Lambda(tf.cast, arguments={'dtype': tf.int64})(startId_layer)
        
        initial_softmax = self.language_model(startId_layer)    
        one_softmax = initial_softmax
        
        # by clipping the values, it can accept inputs that go beyong the 
        # unit hypercube
        clipped_layer = Lambda(clip_layer, arguments={'min_value': 0., 'max_value': 1.})(inputs)   #Clip(0., 1.)(input_point)
        
        unfolding_point = clipped_layer
        
        expanded_os = ExpandDims(1)(one_softmax)
        final_softmaxes = expanded_os
        final_tokens = None  # startId_layer
        curDim = 0

        def create_new_token(inputs):
            
            one_softmax, unfolding_point = inputs
            one_softmax = K.expand_dims(one_softmax, axis=1)
            expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
            
            cumsum = K.cumsum(one_softmax, axis=2)
            cumsum = K.squeeze(cumsum, axis=1)
            cumsum_exclusive = tf.cumsum(one_softmax, axis=2, exclusive=True)
            cumsum_exclusive = K.squeeze(cumsum_exclusive, axis=1)

            value_of_interest = tf.concat([expanded_unfolding_point[:, :, curDim]] * self.vocabSize, 1)
            
            # determine the token selected (2 steps: xor and token)
            # differentiable xor (instead of tf.logical_xor)
            token = pzToSymbol_withArgmax(cumsum, cumsum_exclusive, value_of_interest)
            oh_symbol = tf.one_hot(tf.squeeze(tf.cast(token, dtype=tf.int64), axis=1), self.vocabSize)

            #symbol, oh_symbol = pzToSymbol_noArgmax(cumsum, cumsum_exclusive, value_of_interest)
            
            # expand dimensions to be able to perform a proper matrix 
            # multiplication after
            oh_symbol = tf.expand_dims(oh_symbol, axis=1)
            cumsum_exclusive = tf.expand_dims(cumsum_exclusive, axis=1)
            
            # the c_iti value has to be subtracted to the point for the 
            # next round on this dimension                
            c_iti_value = tf.matmul(oh_symbol, cumsum_exclusive, transpose_b=True)
            c_iti_value = tf.squeeze(c_iti_value, axis=1)
            one_hots = dynamic_one_hot(one_softmax, self.latDim, curDim)
            one_hots = tf.squeeze(one_hots, axis=1)
            
            c_iti = c_iti_value * one_hots
            unfolding_point = tf.subtract(unfolding_point, c_iti)
            
            # the p_iti value has to be divided to the point for the next
            # round on this dimension                
            one_hots = dynamic_one_hot(one_softmax, self.latDim, curDim)
            one_hots = tf.squeeze(one_hots, axis=1)
            p_iti_value = tf.matmul(oh_symbol, one_softmax, transpose_b=True)
            p_iti_value = K.squeeze(p_iti_value, axis=1)
            p_iti_and_zeros = p_iti_value * one_hots
            ones = dynamic_ones(one_softmax, self.latDim)
            ones = K.squeeze(ones, axis=1)
            p_iti_plus_ones = tf.add(p_iti_and_zeros, ones)
            p_iti = tf.subtract(p_iti_plus_ones, one_hots)
            
            #eps = .5e-6; unfolding_point = K.clip(unfolding_point, 0 + eps, 1 - eps)  #hack
            unfolding_point = tf.divide(unfolding_point, p_iti)            
            
            return [token, unfolding_point]
        
        # NOTE: since ending on the EOS token would fail for mini-batches, 
        # the algorithm stops at a maxLen when the length of the sentence 
        # is maxLen
        for _ in range(self.max_senLen):                
            
            token, unfolding_point = Lambda(create_new_token)([one_softmax, unfolding_point])
            token.set_shape((None, 1))
            # output = Lambda(create_new_token)([one_softmax, unfolding_point])
            
            if final_tokens == None:
                final_tokens = token
            else:
                final_tokens = Concatenate(axis=1)([final_tokens, token])
            
            # FIXME: optimal way would be to cut the following tensor in order to be         
            # of length max_senLen    
            tokens = Concatenate(axis=1)([final_tokens, startId_layer])
            
            # get the softmax for the next iteration
            one_softmax = self.language_model(tokens)

            expanded_os = ExpandDims(1)(one_softmax)
            final_softmaxes = Concatenate(axis=1)([final_softmaxes, expanded_os])
            
            # NOTE: at each iteration, change the dimension
            curDim += 1
            if curDim >= self.latDim:
                curDim = 0
        
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




class Differentiable_AriEL(object):

    def __init__(self,
                 vocabSize=5,
                 embDim=2,
                 latDim=3,
                 language_model=None,
                 startId=None,
                 max_senLen=10,
                 output_type='both'):

        self.__dict__.update(vocabSize=vocabSize,
                             latDim=latDim,
                             embDim=embDim,
                             language_model=language_model,
                             startId=startId,
                             max_senLen=max_senLen,
                             output_type=output_type)
        
        # both the encoder and the decoder will share the RNN and the embedding
        # layer
        # tf.reset_default_graph()
        
        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)
            
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
        
        # FIXME: clarify what to do with the padding and EOS
        # vocabSize + 1 for the keras padding + 1 for EOS        
        self.DAriA_encoder = DAriEL_Encoder_Layer(vocabSize=self.vocabSize,
                                                  embDim=self.embDim,
                                                  latDim=self.latDim,
                                                  language_model=self.language_model,
                                                  max_senLen=self.max_senLen,
                                                  startId=self.startId)
        
        self.DAriA_decoder = DAriEL_Decoder_Layer(vocabSize=self.vocabSize,
                                                  embDim=self.embDim,
                                                  latDim=self.latDim,
                                                  max_senLen=self.max_senLen,
                                                  language_model=self.language_model,
                                                  startId=self.startId,
                                                  output_type=self.output_type)
        
    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.DAriA_encoder(input_discrete_seq)
            
    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer        
        return self.DAriA_decoder(input_continuous_point)
        


          
def random_sequences_and_points(batchSize=3, latDim=4, max_senLen=6, repeated=False, vocabSize=3):
    
    if not repeated:
        questions = []
        points = np.random.rand(batchSize, latDim)
        for _ in range(batchSize):
            sentence_length = max_senLen #np.random.choice(max_senLen)
            randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
            #EOS = (vocabSize+1)*np.ones(1)
            #randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, latDim)
        sentence_length = max_senLen #np.random.choice(max_senLen)
        question = np.random.choice(vocabSize, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats = [batchSize], axis=0)
        questions = np.repeat(question, repeats = [batchSize], axis=0)
        
    padded_questions = pad_sequences(questions)
    return padded_questions, points


def test():
    
    max_senLen = 400  #20 #
    vocabSize = 4000  #1500 #
    embDim = int(np.sqrt(vocabSize) + 1)
    latDim = 20
    epochs = 100
    
    questions, _ = random_sequences_and_points(batchSize=10, latDim=latDim, max_senLen=max_senLen, vocabSize=vocabSize)
    answers = to_categorical(questions[:,1], vocabSize)
    print(answers)
    LM = predefined_model(vocabSize, embDim)    
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])    
    LM.fit(questions, answers, epochs=epochs)    
    
    DAriA = Differentiable_AriEL(vocabSize = vocabSize,
                                 embDim = embDim,
                                 latDim = latDim,
                                 max_senLen = max_senLen,
                                 output_type = 'both',
                                 language_model=LM,
                                 startId=0)
    
    input_questions = Input(shape=(latDim,), name='question')    
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point[0])

    _, points = random_sequences_and_points(batchSize=100, latDim=latDim, max_senLen=max_senLen)
    pred = decoder_model.predict(points, verbose=1)
    
    print(pred)



def showGradientsAndTrainableParams(model):
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  

    print("""
          Number of trainable params
          
          """)

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize, suppress=True, precision=3)

def test_detail():
    batch_size = 20
    latDim, curDim, vocabSize = 4, 2, 7
    print('Parameters:\nlatDim = {}\ncurDim = {}\nvocabSize = {}\nbatch_size = {}\n\n'.format(latDim, curDim, vocabSize, batch_size))
    
    point_placeholder = tf.placeholder(tf.float32, shape=(batch_size, latDim))
    softmax_placeholder = tf.placeholder(tf.float32, shape=(batch_size, vocabSize))
    
    one_softmax, unfolding_point = softmax_placeholder, point_placeholder
    one_softmax = K.expand_dims(one_softmax, axis=1)
    expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
    
    cumsum = K.cumsum(one_softmax, axis=2)
    cumsum = K.squeeze(cumsum, axis=1)
    cumsum_exclusive = tf.cumsum(one_softmax, axis=2, exclusive=True)
    cumsum_exclusive = K.squeeze(cumsum_exclusive, axis=1)
    
    value_of_interest = tf.concat([expanded_unfolding_point[:, :, curDim]] * vocabSize, 1)
    
    # argmax approximation
    token = pzToSymbol_withArgmax(cumsum, cumsum_exclusive, value_of_interest)
    xor = tf.one_hot(tf.squeeze(token, axis=1), vocabSize)
    #token, xor = pzToSymbol_noArgmax(cumsum, cumsum_exclusive, value_of_interest)
        
    # expand dimensions to be able to perform a proper matrix 
    # multiplication after
    xor = tf.expand_dims(xor, axis=1)
    cumsum_exclusive = tf.expand_dims(cumsum_exclusive, axis=1)
    
    # the c_iti value has to be subtracted to the point for the 
    # next round on this dimension                
    c_iti_value = tf.matmul(xor, cumsum_exclusive, transpose_b=True)
    c_iti_value = tf.squeeze(c_iti_value, axis=1)
    one_hots = dynamic_one_hot(one_softmax, latDim, curDim)
    one_hots = tf.squeeze(one_hots, axis=1)
    
    c_iti = c_iti_value * one_hots
    unfolding_point = tf.subtract(unfolding_point, c_iti)
    
    # the p_iti value has to be divided to the point for the next
    # round on this dimension                
    one_hots = dynamic_one_hot(one_softmax, latDim, curDim)
    one_hots = tf.squeeze(one_hots, axis=1)
    p_iti_value = tf.matmul(xor, one_softmax, transpose_b=True)
    p_iti_value = K.squeeze(p_iti_value, axis=1)
    p_iti_and_zeros = p_iti_value * one_hots
    ones = dynamic_ones(one_softmax, latDim)
    ones = K.squeeze(ones, axis=1)
    p_iti_plus_ones = tf.add(p_iti_and_zeros, ones)
    p_iti = tf.subtract(p_iti_plus_ones, one_hots)
    
    #eps = 1e-3; unfolding_point = K.clip(unfolding_point, 0 + eps, 1 - eps)  #hack
    unfolding_point = tf.divide(unfolding_point, p_iti)
    
    
    # run
    sess = tf.Session()
    
    random_4softmax = np.random.rand(batch_size, vocabSize)
    
    lines = []
    for line in random_4softmax:
        index = np.random.choice(vocabSize)
        line[index] = 100
        lines.append(line)
    
    random_4softmax = np.array(lines)
    sum_r = random_4softmax.sum(axis=1, keepdims=True)
    initial_softmax = random_4softmax/sum_r
    initial_point = np.random.rand(batch_size, latDim)
    feed_data = {softmax_placeholder: initial_softmax, point_placeholder: initial_point}
    results = sess.run([token, softmax_placeholder, unfolding_point, xor], feed_data)
    
    
    from prettytable import PrettyTable
    t = PrettyTable()
    for a in zip(*results):
        t.add_row([*a])
    
    print(t)

    print("""
          Test Gradients
          
          """)
    
    print('initial_softmax: ', initial_softmax.shape)
    grad = sess.run(tf.gradients(xs=[point_placeholder, softmax_placeholder], ys=token), feed_data)  #[token, unfolding_point]), feed_data)
    for g, w in zip(grad, initial_point):
        print(w)
        print('        g:       ', g)
        print('        g.shape: ', g.shape)
        
        
    print("""
          More Defects to Correct
          
          """)
    
    print(results[0].shape)
    
    
if __name__ == '__main__':    
    #test_detail()
    test()
    
