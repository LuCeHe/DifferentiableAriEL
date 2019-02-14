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
3. every node the same number of tokens
4. start with a number of tokens larger than necessary, and 
     - assign tokens to characters them upon visit, first come first served
5. probably I need a <START> and an <END> tokens

"""

import os
import logging
import numpy as np
from nltk import CFG
import scipy.linalg
import pickle
from decimal import Decimal, localcontext, Context

from nlp import GrammarLanguageModel, Vocabulary, addEndTokenToGrammar, \
                NltkGrammarSampler, tokenize

from AriEL import SentenceEncoder, SentenceDecoder, SentenceEmbedding
from sentenceGenerators import _charactersNumsGenerator

import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                         Bidirectional, LSTM, Lambda, TimeDistributed, \
                         RepeatVector, Activation
from keras.preprocessing.sequence import pad_sequences


from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(2)


logger = logging.getLogger(__name__)






# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> NP VP | NP V
                         VP -> V NP
                         PP -> P NP
                         NP -> Det N
                         Det -> 'a' | 'the'
                         N -> 'dog' | 'cat'
                         V -> 'chased' | 'sat'
                         P -> 'on' | 'in'
                         """)



class vAriEL(SentenceEmbedding):

    def __init__(self, grammar, 
                 ndim=1, 
                 precision=np.finfo(np.float32).precision, 
                 dtype=np.float32, 
                 nbMaxTokens=100, 
                 transform=None, 
                 flexibleBounds=False,
                 name='embedding_arithmetic'):
        self.name = name
            
        vocabulary = Vocabulary.fromGrammar(self.grammarInputAgent)
        super(vAriEL, self).__init__(vocabulary, ndim)

        if precision > np.finfo(dtype).precision:
            logger.warning('Reducing precision because it is higher than what %s can support (%d > %d): ' % (str(dtype), precision, np.finfo(dtype).precision))
            precision = np.finfo(dtype).precision
        self.__dict__.update(grammar=grammar, precision=precision, dtype=dtype, nbMaxTokens=nbMaxTokens)

        # FIXME: I simplified the code to keep it mentally manageable
        self.transform = None
        self.transformInv = None
            
        # define encoder and decoder
        self.encoder = vAriEL_Encoder(self.grammar, ndim=ndim, precision=precision, dtype=dtype, transform=self.transform)
        self.decoder = 1 #vAriEL_Decoder(self.grammar, ndim=ndim, precision=precision, transform=self.transformInv)

    def getEncoder(self):
        return self.encoder

    def getDecoder(self, languageModel = None):
        return self.decoder







def partial_vAriEL_Encoder_model(vocabSize = 101, embDim = 2):    
    
    input_questions = Input(shape=(None,), name='question')
    embed = Embedding(vocabSize, embDim)(input_questions)
        
    # plug biLSTM    
    lstm = LSTM(vocabSize, return_sequences=True)(embed)    
    softmax = TimeDistributed(Activation('softmax'))(lstm)
    
    # up to here it works
    model = Model(inputs=input_questions, outputs=softmax)
    return model
                 


def dynamic_zeros(x, d):
    batch_size = tf.shape(x)[0]
    return tf.zeros(tf.stack([batch_size, 1, d]))

def dynamic_ones(x, d):
    batch_size = tf.shape(x)[0]
    return tf.ones(tf.stack([batch_size, 1, d]))

def dynamic_one_hot(x, d, pos):
    batch_size = tf.shape(x)[0]
    one_hots = tf.ones(tf.stack([batch_size, 1, d]))*tf.one_hot(pos, d)
    return one_hots


def vAriEL_Encoder_model(vocabSize = 101, embDim = 2, latDim = 4, rnn = None, embedding = None):  
    
    # if the input is a rnn, use that, otherwise use an LSTM
    if rnn == None:
        rnn = LSTM(vocabSize, return_sequences=True)
    if embedding == None:
        embedding = Embedding(vocabSize, embDim)
        
    assert 'return_state' in rnn.get_config()
    assert 'embeddings_initializer' in embedding.get_config()
        
    input_questions = Input(shape=(None,), name='question')
    
    embed = embedding(input_questions)
        
    # FIXME: I think arguments passed this way won't be saved with the model
    # follow instead: https://github.com/keras-team/keras/issues/1879
    RNN_starter = Lambda(dynamic_zeros, arguments={'d': embDim})(embed)

    # a zero vector is concatenated as the first word embedding 
    # to start running the RNN that will follow
    concatenation = concatenate([RNN_starter, embed], axis = 1)
      
    rnn_output = rnn(concatenation)    
    softmax = TimeDistributed(Activation('softmax'))(rnn_output)
    
    probs = probsToPoint(vocabSize, latDim)([softmax, input_questions])
    model = Model(inputs=input_questions, outputs=probs)
    return model




class probsToPoint(object):
    def __init__(self, vocabSize=2, latDim=3):
        #super(vAriEL_Encoder, self).__init__()
        self.__dict__.update(vocabSize=vocabSize, latDim=latDim)
    
    def __call__(self, inputs):
        softmax, input_questions = inputs
        
        #assert K.int_shape(softmax)[1] == K.int_shape(input_questions)[1]
        
        def downTheTree(inputs):
            listProbs, listTokens = inputs
            
            # for the matrix multiplications that follow we are not going to 
            # use the output of the LSTM after the last token has passed
            listProbs = listProbs[:,:-1,:]
            
            cumsums =  tf.cumsum(listProbs, axis = 2, exclusive = True)
            #for p_ij, c_ij, token_i in zip(listProbs, cumsums, listTokens):
            
            listTokens = tf.to_int32(listTokens)
            one_hot = K.one_hot(listTokens, self.vocabSize)
            
            p_iti = K.sum(listProbs*one_hot, axis=2)
            c_iti = K.sum(cumsums*one_hot, axis=2)
            
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
            cp = c_latent*p_cumprod
            
            lowBound = tf.reduce_sum(cp, axis=1)
            
            point = lowBound + p_prod/2

            return point
                
                
        pointLatentDim = Lambda(downTheTree)([softmax, input_questions])
        return pointLatentDim





def vAriEL_Decoder_model(vocabSize = 101, embDim = 2, latDim = 4, max_senLen = 10, rnn=None, embedding=None, output_type='both'):  
    
    # if the input is a rnn, use that, otherwise use an LSTM
    if rnn == None:
        rnn = LSTM(vocabSize, return_sequences=True)
    if embedding == None:
        embedding = Embedding(vocabSize, embDim)
        
    assert 'return_state' in rnn.get_config()
    assert 'embeddings_initializer' in embedding.get_config()
    
    input_point = Input(shape=(latDim,), name='input_point')
        
    # FIXME: I think arguments passed this way won't be saved with the model
    # follow instead: https://github.com/keras-team/keras/issues/1879
    RNN_starter = Lambda(dynamic_zeros, arguments={'d': embDim})(input_point)
    
    lstm_output = rnn(RNN_starter)    
    
    grad = tf.gradients(xs=RNN_starter, ys=lstm_output)
    print('grad:   ', grad)
    
    first_softmax = TimeDistributed(Activation('softmax'))(lstm_output)    
    
    grad = tf.gradients(xs=RNN_starter, ys=first_softmax)
    print('grad:   ', grad)
    
    probs = pointToProbs(vocabSize, latDim, embDim, max_senLen, rnn=rnn, embedding=embedding, output_type=output_type)([first_softmax, input_point])

    grad = tf.gradients(xs=input_point, ys=probs)
    print('grad:   ', grad)
    grad = tf.gradients(xs=RNN_starter, ys=probs)
    print('grad:   ', grad)
    
    model = Model(inputs=input_point, outputs=probs)
    return model




class pointToProbs(object):
    def __init__(self, vocabSize=2, latDim=3, embDim=2, max_senLen=10, rnn=None, embedding=None, output_type = 'both'):
        """        
        inputs:
            output_type: 'tokens', 'softmaxes' or 'both'
        """
        
        
        #super(vAriEL_Encoder, self).__init__()
        self.__dict__.update(vocabSize=vocabSize, latDim=latDim, 
                             embDim=embDim, max_senLen=max_senLen, 
                             rnn=rnn, embedding=embedding, output_type=output_type)
    
    def __call__(self, inputs):
        first_softmax, input_point = inputs

        
        def upTheTree(inputs):
            one_softmax, input_point = inputs
            
            unfolding_point = input_point
            
            final_softmaxes = one_softmax
            final_tokens = None
            curDim = 0
            # NOTE: since ending on the EOS token would fail for mini-batches, 
            # the algorithm stops at a maxLen when the length of the sentence 
            # is maxLen
            for _ in range(self.max_senLen):
            
                cumsum = K.cumsum(one_softmax, axis=2)
                cumsum = K.squeeze(cumsum, axis=1)
                cumsum_exclusive = tf.cumsum(one_softmax, axis=2, exclusive = True)
                cumsum_exclusive = K.squeeze(cumsum_exclusive, axis=1)
                
                #print(K.int_shape(unfolding_point))
                expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
                value_of_interest = tf.concat([expanded_unfolding_point[:, :, curDim]]*self.vocabSize, 1)
                
                larger = tf.greater(cumsum, value_of_interest)
                larger_exclusive = tf.greater(cumsum_exclusive, value_of_interest)
                
                
                # determine the token selected
                xor = tf.logical_xor(larger, larger_exclusive)
                xor = tf.cast(xor, tf.float32)                
                token = K.argmax(xor, axis=1)
                token = tf.expand_dims(token, axis=1)
                
                # expand dimensions to be able to performa a proper matrix 
                # multiplication after
                xor = tf.expand_dims(xor, axis=1)
                cumsum_exclusive = tf.expand_dims(cumsum_exclusive, axis=1)           
                
                # the c_iti value has to be subtracted to the point for the 
                # next round on this dimension                
                c_iti_value = tf.matmul(xor, cumsum_exclusive, transpose_b=True)
                c_iti_value = tf.squeeze(c_iti_value, axis=1)
                one_hots = Lambda(dynamic_one_hot, arguments={'d': self.latDim, 'pos': curDim})(first_softmax)
                one_hots = tf.squeeze(one_hots, axis=1)
                
                c_iti = c_iti_value*one_hots
                
                unfolding_point = tf.subtract(unfolding_point, c_iti)
                
                # the p_iti value has to be divided to the point for the next
                # round on this dimension                
                one_hots = Lambda(dynamic_one_hot, arguments={'d': self.latDim, 'pos': curDim})(first_softmax)
                one_hots = tf.squeeze(one_hots, axis=1)
                p_iti_value = tf.matmul(xor, one_softmax, transpose_b=True)
                p_iti_value = K.squeeze(p_iti_value, axis=1)
                p_iti_and_zeros = p_iti_value*one_hots
                ones = Lambda(dynamic_ones, arguments={'d': self.latDim})(first_softmax)
                ones = K.squeeze(ones, axis=1)
                p_iti_plus_ones = tf.add(p_iti_and_zeros, ones)
                p_iti = tf.subtract(p_iti_plus_ones, one_hots)
                
                unfolding_point = tf.divide(unfolding_point, p_iti)
                
                # get the softmax for the next iteration
                embed = self.embedding(token)
                rnn_output = self.rnn(embed)
                one_softmax = TimeDistributed(Activation('softmax'))(rnn_output)
                
                final_softmaxes = tf.concat([final_softmaxes, one_softmax], axis=1, name='concat_softmaxes')
                
                if final_tokens == None:
                    final_tokens = token
                else:
                    final_tokens = tf.concat([final_tokens, token], axis=1, name='concat_tokens')
                
                # NOTE: at each iteration, change the dimension
                curDim += 1
                if curDim >= self.latDim:
                    curDim = 0
            
            # remove last softmax, since the initial was given by the an initial
            # zero vector
            final_softmaxes = final_softmaxes[:,:-1,:]
            
            return  [final_tokens, final_softmaxes]

        # FIXME: give two options: the model giving back the whol softmaxes
        # sequence, or the model giving back the sequence of tokens 
        tokens, softmaxes = Lambda(upTheTree)([first_softmax, input_point])
        
        if self.output_type == 'tokens':
            output = tokens
        elif self.output_type == 'softmaxes':
            output = softmaxes
        elif self.output_type == 'both':
            output = [tokens, softmaxes]
        else:
            raise ValueError('the output_type specified is not implemented!')
        
        return output



        
        
        
def test_vAriEL_AE_cdc_model():
    pass
    



class Differential_AriEL_dcd(object):
    def __init__(self, 
                 vocabSize = 5, 
                 embDim = 2, 
                 latDim = 3, 
                 rnn=None, 
                 embedding=None,
                 max_senLen = 10, 
                 output_type = 'both'):

        self.__dict__.update(vocabSize=vocabSize, 
                             latDim=latDim, 
                             embDim=embDim, 
                             rnn=rnn,
                             embedding=embedding,
                             max_senLen=max_senLen, 
                             output_type='both')
        
        # both the encoder and the decoder will share the RNN and the embedding
        # layer
        tf.reset_default_graph()
        
        # if the input is a rnn, use that, otherwise use an LSTM
        try:
            if 'return_state' in rnn.get_config():
                self.rnn = rnn
        except AttributeError:
            self.rnn = LSTM(vocabSize, return_sequences=True)
        
        if embedding == None:
            embedding = Embedding(vocabSize, embDim)
            
        try:
            assert 'return_state' in self.rnn.get_config()
        except AttributeError:
            raise
        #assert 'return_state' in rnn.get_config()
        assert 'embeddings_initializer' in embedding.get_config()
        
        
        print(self.rnn)
        
        
    def encode(self, input_discrete_seq):
        # FIXME: clarify what to do with the padding and EOS
        # vocabSize + 1 for the keras padding + 1 for EOS
        DAriA_encoder = vAriEL_Encoder_model(vocabSize = self.vocabSize, 
                                             embDim = self.embDim, 
                                             latDim = self.latDim,
                                             rnn = self.rnn,
                                             embedding = self.embedding)

        # it doesn't return a keras Model, it returns a keras Layer
        return DAriA_encoder(input_discrete_seq)
            
    def decode(self, input_continuous_point):
        # FIXME: clarify what to do with the padding and EOS
        # vocabSize + 1 for the keras padding + 1 for EOS
        DAriA_decoder = vAriEL_Decoder_model(vocabSize = self.vocabSize, 
                                             embDim = self.embDim, 
                                             latDim = self.latDim,
                                             rnn = self.rnn,
                                             embedding = self.embedding,
                                             output_type = self.output_type)

        # it doesn't return a keras Model, it returns a keras Layer        
        return DAriA_decoder(input_continuous_point)
    
    def ae(self):
        input_question = Input(shape=(None,), name='discrete_sequence')
        
        DAriA_encoder = vAriEL_Encoder_model(vocabSize = self.vocabSize, 
                                             embDim = self.embDim, 
                                             latDim = self.latDim,
                                             rnn = self.rnn,
                                             embedding = self.embedding)

        # FIXME: clarify what to do with the padding and EOS
        # vocabSize + 1 for the keras padding + 1 for EOS
        DAriA_decoder = vAriEL_Decoder_model(vocabSize = self.vocabSize, 
                                             embDim = self.embDim, 
                                             latDim = self.latDim,
                                             rnn = self.rnn,
                                             embedding = self.embedding,
                                             output_type = self.output_type)
        
    
        
        outputs = DAriA_decoder(DAriA_encoder(input_question))
        
        ae = Model(inputs = input_question, outputs = outputs)
        
        return ae
        
        
        
        
def test_vAriEL_AE_dcd_model():
    vocabSize = 2
    max_senLen = 10
    batchSize = 3
    embDim = 2
    latDim = 5
    
    questions = []
    for _ in range(batchSize):
        sentence_length = np.random.choice(max_senLen)
        randomQ = np.random.choice(vocabSize, sentence_length)
        #EOS = (vocabSize+1)*np.ones(1)        #randomQ = np.concatenate((randomQ, EOS))
        questions.append(randomQ)
        
    padded_questions = pad_sequences(questions)
        
    print(questions)
    print('')
    print(padded_questions)
    print('\n')     
    
    
    
    DAriA_dcd = Differential_AriEL_dcd(vocabSize = vocabSize,
                                       embDim = embDim,
                                       latDim = latDim)


    #input_question = Input(shape=(None,), name='discrete_sequence')
    #print(K.int_shape(input_question))
    #print(input_question.dtype)
    #continuous_latent_space = DAriA_dcd.encode(input_question)
    #print(K.int_shape(continuous_latent_space))
    #print(continuous_latent_space.dtype)
    # in between some neural operations can be defined
    #discrete_output = DAriA_dcd.decode(continuous_latent_space)
    #print(K.int_shape(discrete_output[0]), K.int_shape(discrete_output[1]))
    #print(discrete_output[0].dtype, discrete_output[1].dtype)
    #print('')
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    #model = Model(inputs=input_question, outputs=discrete_output + [continuous_latent_space])
    
    model = DAriA_dcd.ae()
    model.summary()
    print('')
    
    #print(partialModel.predict(question)[0])
    for layer in model.predict(padded_questions):
        print(layer)
        print('\n')


    
    
def simple_tests():
    lstm = LSTM(12)
    print('return_state' in lstm.get_config())
    
    dense = Dense(10)
    
    inputs1 = Input((3,))
    dense1 = dense(inputs1)
    outputs1 = Dense(5)(dense1)
    model1 = Model(inputs=inputs1, outputs=outputs1)
    model1.summary()
    print('-------------------------------------------------------------------------------------------------')
    
    
    inputs2 = Input((5,))
    dense_i = Dense(3)(inputs2)
    dense2 = dense(dense_i)
    outputs2 = Dense(2)(dense2)
    model2 = Model(inputs=inputs2, outputs=outputs2)
    model2.summary()
    print('-------------------------------------------------------------------------------------------------')
    
    
    inputs3 = Input((3,))
    outputs3 = model2(model1(inputs3))
    model3 = Model(inputs=inputs3, outputs=outputs3)
    model3.summary()
    print('-------------------------------------------------------------------------------------------------')
    
    
    output = model3.predict(np.random.rand(2,3))
    print(output)
    
    
    
if __name__ == '__main__':

    #test_vAriEL_Encoder_model()
    #test_vAriEL_Decoder_model()
    test_vAriEL_AE_dcd_model()
    #simple_tests()
    
    
    # FIXME: first token of the question, to make the first siftmax appear
    # hide it inside the code, so the user can simply plug a sentence to the model