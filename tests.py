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
                         RepeatVector, Activation
import tensorflow as tf



vocabSize = 3
max_senLen = 4
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

    # TODO
    # 4. gradient doesn't pass through Embedding or LSTM
    
    print("""
          Test Encoding
          
          """)        

    #partialModel = partial_vAriEL_Encoder_model(vocabSize = 4, embDim = 2)
    
    questions, points = random_sequences_and_points()
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Encoder_model(vocabSize = vocabSize, embDim = 2, latDim = latDim)
    #print(partialModel.predict(question)[0])
    for layer in model.predict(questions):
        print(layer.shape)
        print('')
        print('')

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  

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
    
    # TODO
    # 4. gradient doesn't pass through Embedding or LSTM

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
    #print(partialModel.predict(question)[0])
    for layer in model.predict(points):
        print(layer)
        print('')
        

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, questions)    

   
    
    
    
        
        
def test_vAriEL_AE_dcd_model():
    
    questions, _ = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder
          
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
    #model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    print('')
    print(questions)
    print('')
    print('')
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions)    

    
    
    

        
        
def test_vAriEL_AE_cdc_model():
    
    _, points = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder
          
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
        print(w)
        print('        ', g)  

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, points)    





if __name__ == '__main__':
    #test_vAriEL_Decoder_model()
    print('=========================================================================================')
    #test_vAriEL_Encoder_model()
    print('=========================================================================================')    
    #test_vAriEL_AE_dcd_model()
    print('=========================================================================================')    
    test_vAriEL_AE_cdc_model()