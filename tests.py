#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 23:10:07 2019

@author: perfect
"""

import numpy as np
from vAriEL import vAriEL_Encoder_model, vAriEL_Decoder_model, Differential_AriEL
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                         Bidirectional, LSTM, Lambda, TimeDistributed, \
                         RepeatVector, Activation, GaussianNoise
import keras.backend as K
from keras.legacy import interfaces
import tensorflow as tf


# TODO: move to unittest type of test


vocabSize = 5
max_senLen = 6
batchSize = 2
latDim = 4
embDim = 2
                                
                                
def random_sequences_and_points():
    
    questions = []
    points = np.random.rand(batchSize, latDim)
    for _ in range(batchSize):
        sentence_length = max_senLen #np.random.choice(max_senLen)
        randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
        #EOS = (vocabSize+1)*np.ones(1)
        #randomQ = np.concatenate((randomQ, EOS))
        questions.append(randomQ)
        
    padded_questions = pad_sequences(questions)
    print(questions)
    print('')
    print(padded_questions)
    print('')
    print(points)
    print('')
    print('')

    return padded_questions, points


def test_vAriEL_Encoder_model():
    
    # CHECKED
    # 1. random numpy arrays pass through the encoder succesfully
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Encoding
          
          """)        

    
    questions, points = random_sequences_and_points()
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Encoder_model(vocabSize = vocabSize, embDim = embDim, latDim = latDim)
    #print(partialModel.predict(question)[0])
    for layer in model.predict(questions):
        #print(layer.shape)
        assert layer.shape[0] == latDim
        #print('')
        #print('')

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        #print(w)
        #print('        ', g)  
        if not isinstance(g, tf.IndexedSlices):
            assert g[0] != None
    print("""
          Test fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, points)    
        
def test_vAriEL_Decoder_model():
    
    # CHECKED
    # 1. random numpy arrays pass through the encoder succesfully
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    
    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim, 
                                 max_senLen = max_senLen, 
                                 output_type='tokens')
    
    prediction = model.predict(points)
    
    # The batch size of predicted tokens should contain different sequences of tokens
    assert np.any(prediction[0] != prediction[1])
    

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        #print(w)
        #print('        ', g)  
        assert g[0] != None
    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, questions)    

   
    
    
    
        
        
def test_vAriEL_AE_dcd_model():
    
    questions, _ = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions)    

    
    
    

        
        
def test_vAriEL_AE_cdc_model():
    
    _, points = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder CDC
          
          """)        

    DAriA_cdc = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_point = Input(shape=(latDim,), name='discrete_sequence')
    discrete_output = DAriA_cdc.decode(input_point)
    # in between some neural operations can be defined
    continuous_output = DAriA_cdc.encode(discrete_output)
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_point, outputs=continuous_output)   # + [continuous_latent_space])    
    #model.summary()
    
    for layer in model.predict(points):
        print(layer)
        print('\n')
        
    print('')
    print(points)
    print('')
    print('')
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        #print(w)
        #print('        ', g)
        assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, points)    




class testActiveGaussianNoise(GaussianNoise):
    @interfaces.legacy_gaussiannoise_support
    def __init__(self, stddev, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return noised

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
        
def test_stuff_inside_AE():
    
    
    questions, _ = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    #continuous_latent_space = Dense(latDim)(continuous_latent_space)
    continuous_latent_space = GaussianNoise(stddev=.5)(continuous_latent_space) # WORKS!
    #continuous_latent_space = testActiveGaussianNoise(stddev=5)(continuous_latent_space) # WORKS!

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions, epochs=2)    



    

if __name__ == '__main__':
    #test_vAriEL_Decoder_model()
    print('=========================================================================================')
    #test_vAriEL_Encoder_model()
    print('=========================================================================================')    
    #test_vAriEL_AE_dcd_model()
    print('=========================================================================================')    
    #test_vAriEL_AE_cdc_model()
    print('=========================================================================================')    
    test_stuff_inside_AE()