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
from tf_helpers import slice_, dynamic_ones, dynamic_one_hot, onehot_pseudoD,\
    pzToSymbol_withArgmax, clip_layer, dynamic_fill, dynamic_zeros,\
    pzToSymbolAndZ
from keras_layers import ExpandDims, Slice
tf.compat.v1.disable_eager_execution()
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
tf.random.set_seed(2)

logger = logging.getLogger(__name__)




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

    
def predefined_model(vocabSize, embDim):
    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(128, return_sequences=False)
    
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
    
    max_senLen = 10  #20 #
    vocabSize = 100  #1500 #
    embDim = int(np.sqrt(vocabSize) + 1)
    latDim = 5
    epochs = 10
    
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
    
    showGradientsAndTrainableParams(decoder_model)


def mini_test():
    
    
    batch_size = 2
    latDim, curDim, vocabSize = 4, 2, 7
    print('Parameters:\nlatDim = {}\ncurDim = {}\nvocabSize = {}\nbatch_size = {}\n\n'.format(latDim, curDim, vocabSize, batch_size))
    
    unfolding_point = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, latDim))
    softmax_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, vocabSize))
    t_latDim = tf.shape(unfolding_point)[-1]
    
    
    expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
    t_curDim = tf.constant(curDim)

    x = expanded_unfolding_point[:, :, curDim]
    t_x = expanded_unfolding_point[:, :, t_curDim]  # works!

    one_hots = dynamic_one_hot(softmax_placeholder, latDim, curDim)
    t_one_hots = dynamic_one_hot(softmax_placeholder, t_latDim, t_curDim)

    # run
    sess = tf.compat.v1.Session()
    
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
    feed_data = {softmax_placeholder: initial_softmax, unfolding_point: initial_point}
    results = sess.run([softmax_placeholder, one_hots, t_one_hots], feed_data)
    
    for r in results:
        print(r)
        print('\n\n')
    
if __name__ == '__main__':    
    #test_detail()
    test()
    #mini_test()
    
