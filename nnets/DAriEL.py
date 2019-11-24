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
from DifferentiableAriEL.tf_helpers import slice_, dynamic_ones, dynamic_one_hot, onehot_pseudoD, \
    pzToSymbol_withArgmax, clip_layer, dynamic_fill, dynamic_zeros, \
    pzToSymbolAndZ
from DifferentiableAriEL.keras_layers import ExpandDims, Slice
tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Input, Embedding, \
                         LSTM, Lambda, TimeDistributed, RepeatVector, \
                         Activation, Concatenate, Dense, RNN, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework import function

from numpy.random import seed

seed(3)
tf.set_random_seed(2)

logger = logging.getLogger(__name__)


def showGradientsAndTrainableParams(model):
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
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

    
def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(512, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocabSize, activation='softmax')(lstm_output)
    
    return Model(inputs=input_question, outputs=softmax)


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
        #if not self.startId == 0: raise ValueError('Currently the model works only for startId == 0 ;) ')
        
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
        # startId_layer = Lambda(tf.cast, arguments={'dtype': tf.int64})(startId_layer)
        
        initial_softmax = self.language_model(startId_layer)    
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
            tokens = Concatenate(axis=1)([final_tokens, startId_layer])
            
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
            sentence_length = max_senLen  # np.random.choice(max_senLen)
            randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
            # EOS = (vocabSize+1)*np.ones(1)
            # randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, latDim)
        sentence_length = max_senLen  # np.random.choice(max_senLen)
        question = np.random.choice(vocabSize, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats=[batchSize], axis=0)
        questions = np.repeat(question, repeats=[batchSize], axis=0)
        
    padded_questions = pad_sequences(questions)
    return padded_questions, points


class DAriEL_Decoder_Layer_2(Layer):

    def __init__(self,
                 vocabSize=101,
                 embDim=2,
                 latDim=4,
                 language_model=None,
                 startId=None,
                 **kwargs):  
        super(DAriEL_Decoder_Layer_2, self).__init__(**kwargs)
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
                             language_model=language_model,
                             startId=startId)
        
        # if the input is a rnn, use that, otherwise use an LSTM        
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)          
        
        if self.startId == None: raise ValueError('Define the startId you are using ;) ')
        
    def build(self, input_shape):        
        super(DAriEL_Decoder_Layer_2, self).build(input_shape)  # Be sure to call this at the end
        
    @property
    def state_size(self):
        return (self.vocabSize,
                1,
                self.latDim,
                1,
                1)

    @property
    def output_size(self):
        return self.vocabSize
    
    def call(self, inputs, state):

        input_point = inputs
        one_softmax, tokens, unfolding_point, curDim, timeStep = state

        # initialization        
        startId_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.startId)})(input_point)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)

        initial_softmax = self.language_model(startId_layer)

        # FIXME: it would be interesting to consider what would happen if we feed different points within
        # a batch
        # zero = tf.zeros_like(timeStep)
        pred = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)
        print(pred)
        unfolding_point = tf.cond(pred, lambda: input_point, lambda: unfolding_point)
        one_softmax = tf.cond(pred, lambda: initial_softmax, lambda: one_softmax)
        tokens = tf.cond(pred, lambda: startId_layer, lambda: tokens, name='tokens')
        
        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None, 1))
        
        # tokens = tf.concat([tokens, token], 1) FIXME
        tokens = token
        
        # get the softmax for the next iteration
        tokens_in = Input(tensor=tokens)
        one_softmax = self.language_model(tokens_in)

        # NOTE: at each iteration, change the dimension, and add a timestep
        latDim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred = tf.reduce_mean(curDim) + 1 > tf.reduce_mean(latDim)  # tf.math.greater_equal(curDim, latDim)    
        curDim = tf.cond(pred, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)
        
        new_state = [one_softmax, tokens, unfolding_point, curDim, timeStep]
        output = one_softmax

        return output, new_state
          

def test():
    
    max_senLen = 10  # 20 #
    vocabSize = 100  # 1500 #
    embDim = int(np.sqrt(vocabSize) + 1)
    latDim = 5
    epochs = 10
    
    questions, _ = random_sequences_and_points(batchSize=10, latDim=latDim, max_senLen=max_senLen, vocabSize=vocabSize)
    answers = to_categorical(questions[:, 1], vocabSize)
    print(answers)
    
    LM = predefined_model(vocabSize, embDim)
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
    LM.fit(questions, answers, epochs=epochs)    
    
    DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                 embDim=embDim,
                                 latDim=latDim,
                                 max_senLen=max_senLen,
                                 output_type='both',
                                 language_model=LM,
                                 startId=0)
    
    input_questions = Input(shape=(latDim,), name='question')    
    dense = Dense(4)(input_questions)
    point = DAriA.decode(dense)
    decoder_model = Model(inputs=input_questions, outputs=point[1])

    _, points = random_sequences_and_points(batchSize=100, latDim=latDim, max_senLen=max_senLen)
    pred = decoder_model.predict(points, verbose=1)
    
    print(pred)
    
    showGradientsAndTrainableParams(decoder_model)


def test_2_old():
    vocabSize = 5  # 1500 #
    embDim = 2
    latDim = 4
    max_length = 10
    epochs = 10
    batchSize = 3
    
    DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                 embDim=embDim,
                                 latDim=latDim,
                                 max_senLen=max_length,
                                 output_type='both',
                                 language_model=None,
                                 startId=0)
    
    input_questions = Input(shape=(latDim,), name='question')    
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point[0])
    
    _, points = random_sequences_and_points(batchSize=batchSize, latDim=latDim, max_senLen=max_length)
    pred_output = decoder_model.predict(points, verbose=1)
    
    print(pred_output)
    # print('\n\n\n\n')
    # print(pred_states)

    
def test_2_tf():
    
    """
    vocabSize = 5  # 1500 #
    embDim = 2
    latDim = 4
    max_length = 10
    epochs = 10
    batchSize = 3
    
    tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
      (0) Invalid argument: slice index 4 of dimension 2 out of bounds.
         [[{{node AriEL_decoder/while/strided_slice_2}}]]
         [[AriEL_decoder/while/model_1/lstm/while/LoopCond/_121]]
      (1) Invalid argument: slice index 4 of dimension 2 out of bounds.
         [[{{node AriEL_decoder/while/strided_slice_2}}]]
    
    vocabSize = 6  # 1500 #
    embDim = 3
    latDim = 5
    max_length = 10
    epochs = 10
    batchSize = 2

    tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
      (0) Invalid argument: slice index 5 of dimension 2 out of bounds.
         [[{{node AriEL_decoder/while/strided_slice_2}}]]
      (1) Invalid argument: slice index 5 of dimension 2 out of bounds.
         [[{{node AriEL_decoder/while/strided_slice_2}}]]
         [[AriEL_decoder/transpose_1/_105]]
    
    """

    vocabSize = 6  # 1500 #
    embDim = 3
    latDim = 16
    max_length = 20
    epochs = 10
    batchSize = 3
    
    cell = DAriEL_Decoder_Layer_2(vocabSize=vocabSize,
                                  embDim=embDim,
                                  latDim=latDim,
                                  language_model=None,
                                  startId=0)
    rnn = RNN([cell], return_sequences=True, return_state=True, name='AriEL_decoder')
    
    input_point = Input(shape=(latDim,), name='question')    
    point = RepeatVector(max_length)(input_point)
    print('\npoint: ', K.int_shape(point))
    sequence = rnn(point)[0]  # [1][1]
    decoder_model = Model(inputs=input_point, outputs=sequence)
    
    _, points = random_sequences_and_points(batchSize=batchSize, latDim=latDim, max_senLen=max_length)
    pred_output = decoder_model.predict(points, batch_size=batchSize, verbose=1)
    
    print(np.argmax(pred_output, axis=2))
    # print('\n\n\n\n')
    # print(pred_states)
    
    # showGradientsAndTrainableParams(decoder_model)


def finetuning():

    vocabSize = 6  # 1500 #
    embDim = 3
    latDim = 3
    max_length = 5
    epochs = 10
    batch_size = 3
    startId = 0

    language_model = predefined_model(vocabSize, embDim)  

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    _, inputs = random_sequences_and_points(batchSize=batch_size, latDim=latDim, max_senLen=max_length)
    
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, latDim))

    curDim = tf.zeros(1)
    timeStep = tf.zeros(1)
    one_softmax, tokens, unfolding_point = tf.zeros([batch_size, vocabSize]), tf.zeros(1), tf.zeros([batch_size, latDim])
    state = one_softmax, tokens, unfolding_point, curDim, timeStep
    
    input_point = inputs_placeholder
    
    for _ in range(max_length):
        
        one_softmax, tokens, unfolding_point, curDim, timeStep = state

        # initialization        
        startId_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(startId)})(input_point)
        startId_layer = Lambda(K.squeeze, arguments={'axis': 1})(startId_layer)

        initial_softmax = language_model(startId_layer)
        
        # FIXME: it would be interesting to consider what would happen if we feed different points within
        # a batch
        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)
        
        unfolding_point = tf.cond(pred_t, lambda: input_point, lambda: unfolding_point)
        one_softmax = tf.cond(pred_t, lambda: initial_softmax, lambda: one_softmax)
        tokens = tf.cond(pred_t, lambda: startId_layer, lambda: tokens, name='tokens')
        
        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None, 1))
    
        # tokens = tf.concat([tokens, token], 1) FIXME
        tokens = token
        
        # get the softmax for the next iteration
        tokens_in = Input(tensor=tokens)
        one_softmax = language_model(tokens_in)        
        
        # NOTE: at each iteration, change the dimension, and add a timestep
        latDim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(latDim)  # tf.math.greater_equal(curDim, latDim)    
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)
        
        output = [one_softmax, curDim, token]
        state = [one_softmax, tokens, unfolding_point, curDim, timeStep]
        # return output, new_state

        feed_data = {inputs_placeholder: inputs}
        results = sess.run([output], feed_data)  # ([output, state], feed_data)
        
        for r in results:
            if isinstance(r, list):
                for i in r :
                    print('\n')
                    print(i.shape)
                    print(i)
            else:
                print(r)
            print('\n\n')
            
        print('------------------------------------------------')
    

if __name__ == '__main__':    


    # test_detail()
    # test()
    # test_2_tf()
    # test_2_old
    # test_2_tf()
    # import timeit
    # print(timeit.timeit(test_2_tf, number=10))
    # print(timeit.timeit(test_2_old, number=10))
    finetuning()
