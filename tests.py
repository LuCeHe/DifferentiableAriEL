#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 23:10:07 2019

@author: perfect
"""

import numpy as np
from vAriEL import vAriEL_Encoder_model, vAriEL_Decoder_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf



vocabSize = 2
max_senLen = 3
batchSize = 3
latDim = 9
embDim = 2
max_senLen = 10
                                
                                
def random_sequences_and_points():
    
    questions = []
    points = np.random.rand(batchSize, latDim)
    for _ in range(batchSize):
        sentence_length = np.random.choice(max_senLen)
        randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
        #EOS = (vocabSize+1)*np.ones(1)
        #randomQ = np.concatenate((randomQ, EOS))
        questions.append(randomQ)
        
    padded_questions = pad_sequences(questions)
    print(questions)
    print('')
    print(padded_questions)
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
    # TODO
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    
    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocabSize = vocabSize, embDim = embDim, latDim = latDim, max_senLen = max_senLen, output_type='tokens')
    #print(partialModel.predict(question)[0])
    for layer in model.predict(points):
        print(layer.shape)
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
    
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    #model.fit(points, questions)    



if __name__ == '__main__':
    test_vAriEL_Decoder_model()
    #print('=========================================================================================')
    #test_vAriEL_Encoder_model()