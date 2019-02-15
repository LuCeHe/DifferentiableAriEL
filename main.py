#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:54:15 2019

@author: perfect
"""

# learn language in the AE

import numpy as np
from nltk import CFG
from vAriEL import vAriEL_Encoder_model, vAriEL_Decoder_model, Differential_AriEL
from sentenceGenerators import c2n_generator
from keras.models import Model
from keras.layers import Input





# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> NP VP | NP V
                         VP -> V NP
                         PP -> P NP
                         NP -> Det N
                         Det -> 'a' | 'the'
                         N -> 'dog' | 'cat'
                         V -> 'chased' | 'sat'
                         P -> 'on' | 'in'
                         """)


vocabSize = 3
max_senLen = 48
batchSize = 5
latDim = 5
embDim = 10
epochs = 5
latentTestRate =  2


def main():

    
    # dataset to be learned
    generator_class = c2n_generator(grammar, batchSize, maxlen=max_senLen)
    generator = generator_class.generator()

    vocabSize = generator_class.vocabSize    
    
    print("""
          Test Auto-Encoder on Grammar
          
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
    ae_model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])        
    ae_model.compile(loss='mean_squared_error', optimizer='sgd')
    
    # reuse decoder to define a model to test generation capacity
    input_point = Input(shape=(latDim,), name='continuous_input')
    discrete_output = DAriA_dcd.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=discrete_output)

    
    for epoch in range(epochs):
        sentences = next(generator)

        ae_model.fit(sentences, sentences)    
        
        # FIXME: noise in the latent rep
        if epoch%latentTestRate:
            print("""
                  test ae
                  
                  """)
            indicess = ae_model.predict(sentences)
            sentences_reconstructed = generator_class.indicesToSentences(indicess)
            print(indicess)
            print(sentences_reconstructed)
            print("""
                  test decoder
                  
                  """)
            noise = np.random.rand(batchSize, latDim)
            indicess = decoder_model.predict(noise)
            sentences_generated = generator_class.indicesToSentences(indicess)
            print(sentences_generated)
            





if __name__ == '__main__':
    main()

