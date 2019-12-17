#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:18:53 2019

AriEL for New Word Acquisition

"""

import logging

import tensorflow as tf
from numpy.random import seed

from DifferentiableAriEL.nnets.AriEL_decoder import ArielDecoderLayer0, ArielDecoderLayer1, \
    ArielDecoderLayer2
from DifferentiableAriEL.nnets.AriEL_encoder import ArielEncoderLayer0, ArielEncoderLayer1, \
    ArielEncoderLayer2
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
                 vocab_size=5,
                 emb_dim=2,
                 lat_dim=3,
                 language_model=None,
                 PAD=None,
                 maxlen=10,
                 decoder_type=1,
                 encoder_type=1,
                 size_lat_dim=10,
                 output_type='both'):

        self.common_kwargs = dict()
        self.common_kwargs.update(
            vocab_size=vocab_size,
            lat_dim=lat_dim,
            emb_dim=emb_dim,
            size_lat_dim=size_lat_dim,
            language_model=language_model,
            PAD=PAD,
            maxlen=maxlen)
        self.__dict__.update(**self.common_kwargs, output_type=output_type)

        # if the input is a rnn, use that, otherwise use an LSTM
        if self.language_model == None:
            self.language_model = predefined_model(vocab_size, emb_dim)

        if self.PAD == None: raise ValueError('Define the PAD you are using ;) ')

        if encoder_type == 0:
            self.AriEL_encoder = ArielEncoderLayer0(**self.common_kwargs)
        elif encoder_type == 1:
            # right now this is the better one
            self.AriEL_encoder = ArielEncoderLayer1(**self.common_kwargs)
        elif encoder_type == 2:
            self.AriEL_encoder = ArielEncoderLayer2(**self.common_kwargs)
        else:
            raise NotImplementedError

        if decoder_type == 0:
            # right now this is the better one
            self.AriEL_decoder = ArielDecoderLayer0(**self.common_kwargs, output_type=output_type)
        elif decoder_type == 1:
            self.AriEL_decoder = ArielDecoderLayer1(**self.common_kwargs)
        elif decoder_type == 2:
            self.AriEL_decoder = ArielDecoderLayer2(**self.common_kwargs, output_type=output_type)
        else:
            raise NotImplementedError

    def encode(self, input_discrete_seq):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.AriEL_encoder(input_discrete_seq)

    def decode(self, input_continuous_point):
        # it doesn't return a keras Model, it returns a keras Layer
        return self.AriEL_decoder(input_continuous_point)
