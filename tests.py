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
import tensorflow as tf
from utils import TestActiveGaussianNoise, SelfAdjustingGaussianNoise


# TODO: move to unittest type of test


vocabSize = 6
max_senLen = 6
batchSize = 1 #4
latDim = 5
embDim = 2
                                
                                
def random_sequences_and_points(repeated=False, vocabSize=vocabSize):
    
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
    #print(questions)
    print('')
    print('padded quesitons')
    print(padded_questions)
    print('')
    print('points')
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




    
def test_stuff_inside_AE():
    
    
    questions, _ = random_sequences_and_points(True)
    
    
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

    # Dense WORKS!! (it fits) but loss = 0 even for random initial weights! ERROR!!!!
    #continuous_latent_space = Dense(latDim)(continuous_latent_space)                      
    
    # GaussianNoise(stddev=.02) WORKS!! (it fits) and loss \= 0!! but stddev>=.15 
    # not always :( find reason or if it keeps happening if you restart the kernel
    #continuous_latent_space = GaussianNoise(stddev=.2)(continuous_latent_space) 
    
    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    continuous_latent_space = TestActiveGaussianNoise(stddev=.2)(continuous_latent_space)

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



    



def test_noise_vs_vocabSize(vocabSize=4, std=.2):
    
    
    questions, _ = random_sequences_and_points(False, vocabSize)
    
    
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

    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    #continuous_latent_space = TestActiveGaussianNoise(stddev=std)(continuous_latent_space)
    continuous_latent_space = SelfAdjustingGaussianNoise()(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        


def test_Decoder_forTooMuchNoise():
    """
    it works thanks to value clipping at the entrance of the decoder, other
    solutions might be interesting, like an interplay with layer normalization
    or allowing the latent space to be [-1,1]
    """
    
    print("""
          Test Decoding
          
          """)

    #questions, points = random_sequences_and_points()
    points = 20*np.random.rand(batchSize, latDim)
    
    
    print('points')
    print(points)
    print('')
    
    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim, 
                                 max_senLen = max_senLen, 
                                 output_type='tokens')
    
    prediction = model.predict(points)
    
    print(prediction)    
    


def test_SelfAdjustingGaussianNoise():
    
    print("""
          Test SelfAdjustingGaussianNoise
          
          """)

    ones = np.ones((1,3))
    
    inputs = Input((3,))
    type_sagn = 'scalar'  # 'tensor'  #  'ababas'   #
    output = SelfAdjustingGaussianNoise(type_sagn)(inputs)
    model = Model(inputs, output)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.summary()
    
    # it's learning to decrease its noise so it can map ones to ones
    for _ in range(10):
        model.fit(ones, ones, epochs=100, verbose=0)
        prediction = model.predict(ones)
        print(prediction)    
    


if __name__ == '__main__':
    #test_vAriEL_Decoder_model()
    print('=========================================================================================')
    #test_vAriEL_Encoder_model()
    print('=========================================================================================')    
    #test_vAriEL_AE_dcd_model()
    print('=========================================================================================')    
    #test_vAriEL_AE_cdc_model()
    print('=========================================================================================')    
    #test_stuff_inside_AE()
    print('=========================================================================================')    
    #test_Decoder_forTooMuchNoise()
    print('=========================================================================================')    
    test_SelfAdjustingGaussianNoise()
    

