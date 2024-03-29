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

import logging

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, \
                         Activation, Concatenate, Dense

from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(2)

logger = logging.getLogger(__name__)


def predefined_model(vocab_size, emb_dim):
    embedding = Embedding(vocab_size, emb_dim, mask_zero='True')
    lstm = LSTM(128, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocab_size, activation='softmax')(lstm_output)
    
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

        


def DAriEL_Encoder_model(vocab_size=101,
                         emb_dim=2,
                         lat_dim=4,
                         language_model=None,
                         maxlen=6,
                         startId=None):      
    
    layer = DAriEL_Encoder_Layer(vocab_size=vocab_size, emb_dim=emb_dim,
                                 lat_dim=lat_dim, language_model=language_model,
                                 maxlen=maxlen, startId=startId)
    input_questions = Input(shape=(None,), name='question')    
    point = layer(input_questions)
    model = Model(inputs=input_questions, outputs=point)
    return model


# FIXME: encoder
class DAriEL_Encoder_Layer(object):

    def __init__(self,
                 vocab_size=101,
                 emb_dim=2,
                 lat_dim=4,
                 language_model=None,
                 maxlen=6,
                 startId=None,
                 softmaxes=False):  

        self.__dict__.update(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             language_model=language_model,
                             maxlen=maxlen,
                             startId=startId,
                             softmaxes=softmaxes)
        
        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)
            
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
        if not self.startId == 0: raise ValueError('Currently the model works only for startId == 0 ;) ')
        
    def __call__(self, input_questions):
                
        startId_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.startId)})(input_questions)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)
            
        softmax = self.language_model(startId_layer)
        
        expanded_os = ExpandDims(1)(softmax)
        final_softmaxes = expanded_os
        
        question_segments = []
        for final in range(self.maxlen):
            partial_question = Slice(1, 0, final + 1)(input_questions)   
            question_segments.append(partial_question)
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
            
            listTokens = tf.to_int32(listTokens)
            one_hot = K.one_hot(listTokens, self.vocab_size)
            
            p_iti = K.sum(listProbs * one_hot, axis=2)
            c_iti = K.sum(cumsums * one_hot, axis=2)
            
            # Create another vector containing zeroes to pad `a` to (2 * 3) elements.
            zero_padding = Lambda(dynamic_zeros, arguments={'d': self.lat_dim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(p_iti)
            zero_padding = K.squeeze(zero_padding, axis=1)
            ones_padding = Lambda(dynamic_ones, arguments={'d': self.lat_dim * tf.shape(p_iti)[1] - tf.shape(p_iti)[1]})(p_iti)
            ones_padding = K.squeeze(ones_padding, axis=1)
            
            # Concatenate `a_as_vector` with the padding.
            p_padded = tf.concat([p_iti, ones_padding], 1)
            c_padded = tf.concat([c_iti, zero_padding], 1)
            
            # Reshape the padded vector to the desired shape.
            p_latent = tf.reshape(p_padded, [-1, tf.shape(p_iti)[1], self.lat_dim])
            c_latent = tf.reshape(c_padded, [-1, tf.shape(c_iti)[1], self.lat_dim])
            
            # calculate the final position determined by AriEL
            p_cumprod = tf.cumprod(p_latent, axis=1, exclusive=True)
            p_prod = tf.reduce_prod(p_latent, axis=1)
            cp = c_latent * p_cumprod
            
            lowBound = tf.reduce_sum(cp, axis=1)
            
            point = lowBound + p_prod / 2

            return point
                
        pointLatentDim = Lambda(downTheTree, name='downTheTree')([softmax, input_questions])
        return pointLatentDim


def DAriEL_Decoder_model(vocab_size=101,
                         emb_dim=2,
                         lat_dim=4,
                         maxlen=10,
                         language_model=None,
                         startId=None,
                         output_type='both'):  
    
    layer = DAriEL_Decoder_Layer(vocab_size=vocab_size, emb_dim=emb_dim,
                                 lat_dim=lat_dim, maxlen=maxlen,
                                 language_model=language_model, startId=startId,
                                 output_type=output_type)
    input_point = Input(shape=(lat_dim,), name='input_point')
    discrete_sequence_output = layer(input_point)    
    model = Model(inputs=input_point, outputs=discrete_sequence_output)
    return model


class DAriEL_Decoder_Layer(object):

    def __init__(self,
                 vocab_size=101,
                 emb_dim=2,
                 lat_dim=4,
                 maxlen=10,
                 language_model=None,
                 startId=None,
                 output_type='both'):  
        
        self.__dict__.update(vocab_size=vocab_size,
                             emb_dim=emb_dim,
                             lat_dim=lat_dim,
                             maxlen=maxlen,
                             language_model=language_model,
                             startId=startId,
                             output_type=output_type)
        
        # if the input is a rnn, use that, otherwise use an LSTM
        
        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)
        
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
    
    def __call__(self, input_point):
            
        # FIXME: I think arguments passed this way won't be saved with the model
        # follow instead: https://github.com/keras-team/keras/issues/1879
        # RNN_starter = Lambda(dynamic_zeros, arguments={'d': self.emb_dim})(input_point)
        # RNN_starter = Lambda(dynamic_fill, arguments={'d': self.emb_dim, 'value': .5})(input_point)

        output = pointToProbs(vocab_size=self.vocab_size,
                              lat_dim=self.lat_dim,
                              emb_dim=self.emb_dim,
                              maxlen=self.maxlen,
                              language_model=self.language_model,
                              startId=self.startId,
                              output_type=self.output_type)(input_point)
    
        return output


class pointToProbs(object):

    def __init__(self,
                 vocab_size=2,
                 lat_dim=3,
                 emb_dim=2,
                 maxlen=10,
                 language_model=None,
                 startId=None,
                 output_type='both'):
        """        
        inputs:
            output_type: 'tokens', 'softmaxes' or 'both'
        """
        self.__dict__.update(vocab_size=vocab_size, lat_dim=lat_dim,
                             emb_dim=emb_dim, maxlen=maxlen,
                             language_model=language_model,
                             startId=startId,
                             output_type=output_type)
    
    def __call__(self, inputs):
        input_point = inputs
        
        startId_layer = Lambda(dynamic_fill, arguments={'d': self.maxlen, 'value': float(self.startId)})(input_point)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)
        
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

            value_of_interest = tf.concat([expanded_unfolding_point[:, :, curDim]] * self.vocab_size, 1)
            
            # determine the token selected (2 steps: xor and token)
            # differentiable xor (instead of tf.logical_xor)                
            c_minus_v = tf.subtract(cumsum, value_of_interest)
            ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
            signed_xor = c_minus_v * ce_minus_c
            abs_sx = tf.abs(signed_xor)
            almost_xor = tf.divide(signed_xor, abs_sx)
            almost_xor = tf.add(almost_xor, -1)
            xor = tf.abs(tf.divide(almost_xor, -2))
            
            # differentiable argmax (instead of tf.argmax)                
            almost_token = tf.divide(c_minus_v, tf.abs(c_minus_v))
            almost_token = tf.abs(tf.divide(tf.add(almost_token, -1), -2))
            token = tf.reduce_sum(almost_token, axis=1)
            token = tf.expand_dims(token, axis=1)
            
            # expand dimensions to be able to perform a proper matrix 
            # multiplication after
            xor = tf.expand_dims(xor, axis=1)
            cumsum_exclusive = tf.expand_dims(cumsum_exclusive, axis=1)                   
            
            # the c_iti value has to be subtracted to the point for the 
            # next round on this dimension                
            c_iti_value = tf.matmul(xor, cumsum_exclusive, transpose_b=True)
            c_iti_value = tf.squeeze(c_iti_value, axis=1)
            one_hots = dynamic_one_hot(one_softmax, self.lat_dim, curDim)
            one_hots = tf.squeeze(one_hots, axis=1)
            
            c_iti = c_iti_value * one_hots
            unfolding_point = tf.subtract(unfolding_point, c_iti)
            
            # the p_iti value has to be divided to the point for the next
            # round on this dimension                
            one_hots = dynamic_one_hot(one_softmax, self.lat_dim, curDim)
            one_hots = tf.squeeze(one_hots, axis=1)
            p_iti_value = tf.matmul(xor, one_softmax, transpose_b=True)
            p_iti_value = K.squeeze(p_iti_value, axis=1)
            p_iti_and_zeros = p_iti_value * one_hots
            ones = dynamic_ones(one_softmax, self.lat_dim)
            ones = K.squeeze(ones, axis=1)
            p_iti_plus_ones = tf.add(p_iti_and_zeros, ones)
            p_iti = tf.subtract(p_iti_plus_ones, one_hots)
            
            unfolding_point = tf.divide(unfolding_point, p_iti)            
            
            return [token, unfolding_point]
        
        # NOTE: since ending on the EOS token would fail for mini-batches, 
        # the algorithm stops at a maxLen when the length of the sentence 
        # is maxLen
        for tree_node in range(self.maxlen):
                
            token, unfolding_point = Lambda(create_new_token)([one_softmax, unfolding_point])
            # output = Lambda(create_new_token)([one_softmax, unfolding_point])
            
            if final_tokens == None:
                final_tokens = token
            else:
                final_tokens = Concatenate(axis=1)([final_tokens, token])
            
            # FIXME: optimal way would be to cut the following tensor in order to be         
            # of length maxlen
            tokens = Concatenate(axis=1)([final_tokens, startId_layer])
            
            # get the softmax for the next iteration
            one_softmax = self.language_model(tokens)

            expanded_os = ExpandDims(1)(one_softmax)
            final_softmaxes = Concatenate(axis=1)([final_softmaxes, expanded_os])
            
            # NOTE: at each iteration, change the dimension
            curDim += 1
            if curDim >= self.lat_dim:
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
                 vocab_size=5,
                 emb_dim=2,
                 lat_dim=3,
                 language_model=None,
                 startId=None,
                 maxlen=10,
                 output_type='both'):

        self.__dict__.update(vocab_size=vocab_size,
                             lat_dim=lat_dim,
                             emb_dim=emb_dim,
                             language_model=language_model,
                             startId=startId,
                             maxlen=maxlen,
                             output_type=output_type)
        
        # both the encoder and the decoder will share the RNN and the embedding
        # layer
        # tf.reset_default_graph()
        
        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)
            
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
        
        # FIXME: clarify what to do with the padding and EOS
        # vocab_size + 1 for the keras padding + 1 for EOS
        self.DAriA_encoder = DAriEL_Encoder_Layer(vocab_size=self.vocab_size,
                                                  emb_dim=self.emb_dim,
                                                  lat_dim=self.lat_dim,
                                                  language_model=self.language_model,
                                                  maxlen=self.maxlen,
                                                  startId=self.startId)
        
        self.DAriA_decoder = DAriEL_Decoder_Layer(vocab_size=self.vocab_size,
                                                  emb_dim=self.emb_dim,
                                                  lat_dim=self.lat_dim,
                                                  maxlen=self.maxlen,
                                                  language_model=self.language_model,
                                                  startId=self.startId,
                                                  output_type=self.output_type)
        
    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.DAriA_encoder(input_discrete_seq)
            
    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer        
        return self.DAriA_decoder(input_continuous_point)
        

def test():
    import numpy as np
    sentences = np.array([[0, 0, 1, 2, 2],
                          [0, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1]])
    model = DAriEL_Encoder_model(vocab_size=3, startId=0)
    prediction = model.predict(sentences)
    print(prediction)


if __name__ == '__main__':    
    test()
    
