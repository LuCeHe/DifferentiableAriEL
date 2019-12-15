#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:18:53 2019

AriEL for New Word Acquisition

"""

import logging

import tensorflow as tf
from numpy.random import seed

from DifferentiableAriEL.nnets.AriEL_decoder import DAriEL_Decoder_Layer_0, DAriEL_Decoder_Layer_1, \
    DAriEL_Decoder_Layer_2
from DifferentiableAriEL.nnets.AriEL_encoder import DAriEL_Encoder_Layer_0, DAriEL_Encoder_Layer_1
from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model

tf.compat.v1.disable_eager_execution()

seed(3)
tf.set_random_seed(2)

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class AriEL(object):
    """ both the encoder and the decoder will share the RNN and the embedding
    layer
    """

    def __init__(self,
                 vocabSize=5,
                 embDim=2,
                 latDim=3,
                 language_model=None,
                 PAD=None,
                 max_senLen=10,
                 decoder_type=1,
                 encoder_type=1,
                 size_latDim=10,
                 output_type='both'):

        self.common_kwargs = dict()
        self.common_kwargs.update(
            vocabSize=vocabSize,
            latDim=latDim,
            embDim=embDim,
            size_latDim=size_latDim,
            language_model=language_model,
            PAD=PAD,
            max_senLen=max_senLen,
            output_type=output_type)
        self.__dict__.update(**self.common_kwargs)

        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocabSize, embDim)

        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')

        if encoder_type == 0:
            self.DAriA_encoder = DAriEL_Encoder_Layer_0(**self.common_kwargs)
        elif encoder_type == 1:
            # right now this is the better one
            self.DAriA_encoder = DAriEL_Encoder_Layer_1(**self.common_kwargs)
        else:
            raise NotImplementedError

        if decoder_type == 0:
            # right now this is the better one     
            self.DAriA_decoder = DAriEL_Decoder_Layer_0(**self.common_kwargs)
        elif decoder_type == 1:
            self.DAriA_decoder = DAriEL_Decoder_Layer_1(**self.common_kwargs)
        elif decoder_type == 2:
            self.DAriA_decoder = DAriEL_Decoder_Layer_2(**self.common_kwargs)
        else:
            raise NotImplementedError

    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.DAriA_encoder(input_discrete_seq)

    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer  
        return self.DAriA_decoder(input_continuous_point)
