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
from keras import optimizers




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


vocabSize = 3  # this value is going to be overwriter after the sentences generator
max_senLen = 24
batchSize = 100
latDim = 16
embDim = 10
epochs = 100
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
    
    
    clippedAdam = optimizers.Adam(clipnorm=1.)
    ae_model.compile(loss='mean_squared_error', optimizer=clippedAdam)
    print('')
    ae_model.summary()
    
    # reuse decoder to define a model to test generation capacity
    input_point = Input(shape=(latDim,), name='continuous_input')
    discrete_output = DAriA_dcd.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=discrete_output)
    

    
    for epoch in range(epochs):
        print("""
              fit ae
              
              """)
        indices_sentences = next(generator)
        indicess = ae_model.predict(indices_sentences)
        sentences_reconstructed = generator_class.indicesToSentences(indicess)
        ae_model.fit(indices_sentences, indices_sentences, epochs)    
        
        # FIXME: noise in the latent rep
        if epoch%latentTestRate:
            print("""
                  test ae
                  
                  """)
            print('original sentences')
            print('')
            sentences = generator_class.indicesToSentences(indices_sentences)
            print(sentences)
            indicess = ae_model.predict(indices_sentences)
            sentences_reconstructed = generator_class.indicesToSentences(indicess)
            print('')
            print('original indices')
            print('')
            print(indicess)
            print('')
            print('reconstructed sentences')
            print('')
            print(sentences_reconstructed)
            print("""
                  test decoder
                  
                  """)
            noise = np.random.rand(batchSize, latDim)
            indicess = decoder_model.predict(noise)
            sentences_generated = generator_class.indicesToSentences(indicess)
            print('generated sentences')
            print('')
            print(sentences_generated)
            print('')





if __name__ == '__main__':
    main()

