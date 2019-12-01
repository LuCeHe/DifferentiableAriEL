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


def showGradientsAndTrainableParams(model):
    
    logger.info("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        logger.info(w)
        logger.info('        ', g)  

    logger.info("""
          Number of trainable params
          
          """)

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
    logger.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logger.info('Trainable params: {:,}'.format(trainable_count))
    logger.info('Non-trainable params: {:,}'.format(non_trainable_count))

    
def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(256, return_sequences=False)
    
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
                         PAD=None):      
    
    layer = DAriEL_Encoder_Layer(vocabSize=vocabSize, embDim=embDim,
                                 latDim=latDim, language_model=language_model,
                                 max_senLen=max_senLen, PAD=PAD)        
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


class Differentiable_AriEL(object):

    def __init__(self,
                 vocabSize=5,
                 embDim=2,
                 latDim=3,
                 language_model=None,
                 PAD=None,
                 max_senLen=10,
                 tf_RNN=True,
                 output_type='both'):

        self.__dict__.update(vocabSize=vocabSize,
                             latDim=latDim,
                             embDim=embDim,
                             language_model=language_model,
                             PAD=PAD,
                             max_senLen=max_senLen,
                             tf_RNN=tf_RNN,
                             output_type=output_type)
        
        # both the encoder and the decoder will share the RNN and the embedding
        # layer
        # tf.reset_default_graph()
        
        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)
            
        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')
        
        # FIXME: clarify what to do with the padding and EOS
        # vocabSize + 1 for the keras padding + 1 for EOS        
        self.DAriA_encoder = DAriEL_Encoder_Layer(vocabSize=self.vocabSize,
                                                  embDim=self.embDim,
                                                  latDim=self.latDim,
                                                  language_model=self.language_model,
                                                  max_senLen=self.max_senLen,
                                                  PAD=self.PAD)
        
        self.DAriA_decoder = DAriEL_Decoder_Layer(vocabSize=self.vocabSize,
                                                  embDim=self.embDim,
                                                  latDim=self.latDim,
                                                  max_senLen=self.max_senLen,
                                                  language_model=self.language_model,
                                                  PAD=self.PAD,
                                                  output_type=self.output_type)
        
    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.DAriA_encoder(input_discrete_seq)
            
    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer    
        
        if self.tf_RNN:
            cell = DAriEL_Decoder_Layer_2(vocabSize=self.vocabSize,
                                          embDim=self.embDim,
                                          latDim=self.latDim,
                                          language_model=self.language_model,
                                          PAD=self.PAD)
            rnn = RNN([cell], return_sequences=True, return_state=True, name='AriEL_decoder')
            
            input_point = Input(shape=(self.latDim,), name='question')    
            point = RepeatVector(self.max_senLen)(input_point)
            o_s = rnn(point)  # [1][1]
            decoder_model = Model(inputs=input_point, outputs=o_s)
            return decoder_model(input_continuous_point)
        else:
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
                 PAD=None,
                 **kwargs):  
        super(DAriEL_Decoder_Layer_2, self).__init__(**kwargs)
        
        self.__dict__.update(vocabSize=vocabSize,
                             embDim=embDim,
                             latDim=latDim,
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
        PAD_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(self.PAD)})(input_point)
        PAD_layer = Lambda(K.squeeze, arguments={'axis': 1})(PAD_layer)

        initial_softmax = self.language_model(PAD_layer)
        
        # FIXME: it would be interesting to consider what would happen if we feed different points within
        # a batch
        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)

        unfolding_point = tf.cond(pred_t, lambda: input_point, lambda: unfolding_point)
        one_softmax = tf.cond(pred_t, lambda: initial_softmax, lambda: one_softmax)
        tokens = tf.cond(pred_t, lambda: PAD_layer, lambda: tokens, name='tokens')
        
        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None, 1))
        
        # tokens = tf.concat([tokens, token], 1) FIXME
        tokens = token
        
        # get the softmax for the next iteration
        tokens_in = Input(tensor=tokens)
        one_softmax = self.language_model(tokens_in)

        # NOTE: at each iteration, change the dimension, and add a timestep
        latDim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(latDim)  # tf.math.greater_equal(curDim, latDim)    
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
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
    logger.info(answers)
    
    LM = predefined_model(vocabSize, embDim)
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
    LM.fit(questions, answers, epochs=epochs)    
    
    DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                 embDim=embDim,
                                 latDim=latDim,
                                 max_senLen=max_senLen,
                                 output_type='both',
                                 language_model=LM,
                                 PAD=0)
    
    input_questions = Input(shape=(latDim,), name='question')    
    dense = Dense(4)(input_questions)
    point = DAriA.decode(dense)
    decoder_model = Model(inputs=input_questions, outputs=point[1])

    _, points = random_sequences_and_points(batchSize=100, latDim=latDim, max_senLen=max_senLen)
    pred = decoder_model.predict(points, verbose=1)
    
    logger.info(pred)
    
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
                                 PAD=0)
    
    input_questions = Input(shape=(latDim,), name='question')    
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point[0])
    
    _, points = random_sequences_and_points(batchSize=batchSize, latDim=latDim, max_senLen=max_length)
    pred_output = decoder_model.predict(points, verbose=1)
    
    logger.info(pred_output)

    
def test_2_tf():
    
    vocabSize = 6  # 1500 #
    embDim = 4
    latDim = 2
    max_length = 7
    batchSize = 3
    PAD = 0
    
    DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                 embDim=embDim,
                                 latDim=latDim,
                                 max_senLen=max_length,
                                 output_type='both',
                                 language_model=None,
                                 tf_RNN=True,
                                 PAD=PAD)
    
    input_questions = Input(shape=(latDim,), name='question')    
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point)
    
    _, points = random_sequences_and_points(batchSize=batchSize, latDim=latDim, max_senLen=max_length)
    pred_output = decoder_model.predict(points, batch_size=batchSize, verbose=1)
    
    logger.warn(np.argmax(pred_output, axis=2))


def replace_column(matrix, new_column, r):
    dynamic_index = tf.cast(tf.squeeze(r), dtype=tf.int64)
    num_cols = tf.shape(matrix)[1]
    #new_matrix = tf.assign(matrix[:, dynamic_index], new_column)
    index_row = tf.stack([ tf.eye(num_cols, dtype=tf.float32)[dynamic_index, :] ])
    old_column = matrix[:, dynamic_index]
    new = tf.matmul(tf.stack([new_column], axis=1), index_row)
    old = tf.matmul(tf.stack([old_column], axis=1), index_row)
    new_matrix = (matrix - old) + new
    return new_matrix



def replace_column_test():
    batch_size = 2
    max_length = 3
    
    sess = tf.Session()
    timeStep = tf.cast(tf.squeeze(2*tf.ones(1)), dtype=tf.int64)
    
    tokens = tf.zeros([batch_size, max_length])
    column = tf.ones([batch_size,])
    out_tokens = replace_column(tokens, column, timeStep)
    
    results = sess.run([tokens, out_tokens]) 
    
    for r in results:
        print('\na result')
        print(r)

def finetuning():

    vocabSize = 6  # 1500 #
    embDim = 5
    latDim = 2
    max_length = 7
    batch_size = 4
    PAD = 0

    language_model = predefined_model(vocabSize, embDim)  

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    _, inputs = random_sequences_and_points(batchSize=batch_size, latDim=latDim, max_senLen=max_length)
    
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None, latDim))

    curDim = tf.zeros(1)
    timeStep = tf.zeros(1)
    b = tf.shape(inputs_placeholder)[0]
    one_softmax, unfolding_point = tf.zeros([b, vocabSize]), tf.zeros([b, latDim])
    tokens = tf.zeros([b, max_length])
    state = one_softmax, tokens, unfolding_point, curDim, timeStep
    
    input_point = inputs_placeholder
    
    all_results = []
    for _ in tqdm(range(max_length)):
        
        input_point = input_point
        one_softmax, tokens, unfolding_point, curDim, timeStep = state

        # initialization        
        start_layer = Lambda(dynamic_fill, arguments={'d': 1, 'value': float(PAD)})(input_point)
        start_layer = Lambda(K.squeeze, arguments={'axis': 1})(start_layer)

        initial_softmax = language_model(start_layer)
        
        # FIXME: it would be interesting to consider what would happen if we feed different
        # points within a batch
        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)
        
        unfolding_point = tf.cond(pred_t, lambda: input_point, lambda: unfolding_point)
        one_softmax = tf.cond(pred_t, lambda: initial_softmax, lambda: one_softmax)
        #tokens = tf.cond(pred_t, lambda: start_layer, lambda: tokens, name='tokens')
        
        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None,1))
        token = tf.squeeze(token, axis=1)
    
        # tokens = tf.concat([tokens, token], 1) FIXME
        print(K.int_shape(token))
        print(K.int_shape(tokens))
        tokens = replace_column(tokens, token, timeStep)
        print(K.int_shape(tokens))
        
        # get the softmax for the next iteration
        #tokens_in = Input(tensor=tokens)
        #one_softmax = language_model(tokens_in)        
        
        # NOTE: at each iteration, change the dimension, and add a timestep
        latDim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(latDim)  # tf.math.greater_equal(curDim, latDim)
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)
        
        output = [curDim, timeStep, tokens]
        state = [one_softmax, tokens, unfolding_point, curDim, timeStep]
        # return output, new_state

        feed_data = {inputs_placeholder: inputs}
        results = sess.run(output, feed_data)  # ([output, state], feed_data)
        all_results.append(results)
        
    t = PrettyTable()
    for a in all_results:
        t.add_row([*a])
    
    print(t)
    

if __name__ == '__main__':    

    #replace_column_test()
    finetuning()
    #test_2_tf()
