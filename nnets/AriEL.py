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

from numpy.random import seed
import logging

import tensorflow as tf
from DifferentiableAriEL.nnets.AriEL_encoder import DAriEL_Encoder_Layer_0, DAriEL_Encoder_Layer_1
from DifferentiableAriEL.nnets.AriEL_decoder import DAriEL_Decoder_Layer_0, DAriEL_Decoder_Layer_1, DAriEL_Decoder_Layer_2
from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, RepeatVector, RNN


seed(3)
tf.set_random_seed(2)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)





class AriEL(object):

    def __init__(self,
                 vocabSize=5,
                 embDim=2,
                 latDim=3,
                 language_model=None,
                 PAD=None,
                 max_senLen=10,
                 decoder_type=1,
                 encoder_type=1,
                 size_latDim = 10,
                 output_type='both'):

        self.__dict__.update(vocabSize=vocabSize,
                             latDim=latDim,
                             embDim=embDim,
                             language_model=language_model,
                             PAD=PAD,
                             max_senLen=max_senLen,
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
        
        if encoder_type == 0:
            self.DAriA_encoder = DAriEL_Encoder_Layer_0(vocabSize=self.vocabSize,
                                                        embDim=self.embDim,
                                                        latDim=self.latDim,
                                                        language_model=self.language_model,
                                                        max_senLen=self.max_senLen,
                                                        PAD=self.PAD)
        elif encoder_type == 1:
            # right now this is the better one
            self.DAriA_encoder = DAriEL_Encoder_Layer_1(vocabSize=self.vocabSize,
                                                        embDim=self.embDim,
                                                        latDim=self.latDim,
                                                        language_model=self.language_model,
                                                        max_senLen=self.max_senLen,
                                                        PAD=self.PAD,
                                                        size_latDim = size_latDim)
        else:
            raise NotImplementedError

        if decoder_type == 0:       
            # right now this is the better one     
            self.DAriA_decoder = DAriEL_Decoder_Layer_0(vocabSize=self.vocabSize,
                                                        embDim=self.embDim,
                                                        latDim=self.latDim,
                                                        max_senLen=self.max_senLen,
                                                        language_model=self.language_model,
                                                        PAD=self.PAD,
                                                        size_latDim = size_latDim,
                                                        output_type=self.output_type)
        elif decoder_type == 1:
            cell = DAriEL_Decoder_Layer_1(vocabSize=self.vocabSize,
                                          embDim=self.embDim,
                                          latDim=self.latDim,
                                          max_senLen=self.max_senLen,
                                          language_model=self.language_model,
                                          PAD=self.PAD)
            rnn = RNN([cell], return_sequences=True, return_state=True, name='AriEL_decoder')
            
            input_point = Input(shape=(self.latDim,), name='question')    
            point = RepeatVector(self.max_senLen)(input_point)
            o_s = rnn(point)  
            self.DAriA_decoder = Model(inputs=input_point, outputs=o_s)
        elif decoder_type == 2:
            self.DAriA_decoder = DAriEL_Decoder_Layer_2(vocabSize=self.vocabSize,
                                                        latDim=self.latDim,
                                                        max_senLen=self.max_senLen,
                                                        language_model=self.language_model,
                                                        size_latDim=size_latDim,
                                                        PAD=self.PAD)            
        else:
            raise NotImplementedError

    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.DAriA_encoder(input_discrete_seq)
            
    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer  
        return self.DAriA_decoder(input_continuous_point)




