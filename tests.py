#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 23:10:07 2019

@author: perfect
"""

import numpy as np
from vAriEL import vAriEL_Encoder_model, vAriEL_Decoder_model
from keras.preprocessing.sequence import pad_sequences



def test_vAriEL_Encoder_model():
    #partialModel = partial_vAriEL_Encoder_model(vocabSize = 4, embDim = 2)
    vocabSize = 2
    max_senLen = 3
    batchSize = 3
    latDim = 9
    
    questions = []
    for _ in range(batchSize):
        sentence_length = np.random.choice(max_senLen)
        randomQ = np.random.choice(vocabSize, sentence_length) + 1
        EOS = (vocabSize+1)*np.ones(1)
        randomQ = np.concatenate((randomQ, EOS))
        questions.append(randomQ)
        
    padded_questions = pad_sequences(questions)
    print("""
          Test Encoding
          
          """)        
    print(questions)
    print('')
    print(padded_questions)
    print('')
    print('')
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Encoder_model(vocabSize = vocabSize + 2, embDim = 2, latDim = latDim)
    #print(partialModel.predict(question)[0])
    for layer in model.predict(padded_questions):
        print(layer.shape)
        print('')
        print('')


def test_vAriEL_Decoder_model():
    #partialModel = partial_vAriEL_Encoder_model(vocabSize = 4, embDim = 2)
    vocabSize = 3
    max_senLen = 5
    batchSize = 2
    latDim = 4
    
    
    questions = np.random.rand(batchSize, latDim)
    
    print("""
          Test Decoding
          
          """)
    print(questions)
    print('')
    print('')
    print('-----------------------------------------------------------------------')
    print('')
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocabSize = vocabSize + 2, embDim = 2, latDim = latDim, max_senLen = max_senLen, output_type='tokens')
    #print(partialModel.predict(question)[0])
    for layer in model.predict(questions):
        print(layer.shape)
        print('')
        



if __name__ == '__main__':
    test_vAriEL_Decoder_model()
    print('=========================================================================================')
    test_vAriEL_Encoder_model()