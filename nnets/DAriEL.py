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
from nnets.AriEL_encoder import DAriEL_Encoder_Layer
from nnets.AriEL_decoder import DAriEL_Decoder_Layer, DAriEL_Decoder_Layer_2

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
                                          max_senLen=self.max_senLen,
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




    

if __name__ == '__main__':    

    #replace_column_test()
    #finetuning()
    test_2_tf()
