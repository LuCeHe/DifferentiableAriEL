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


# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> 'ABC' | 'AAC' | 'BA'
                         """)


vocabSize = 3  # this value is going to be overwriter after the sentences generator
max_senLen = 6 #24
batchSize = 128
latDim = 7
embDim = 5
epochs = 100
epochs_in = 20
latentTestRate = 10
    
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
                                   output_type = 'both')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    ae_model = Model(inputs=input_question, outputs=discrete_output[0])   # + [continuous_latent_space])        
    
    
    clippedAdam = optimizers.Adam(lr=2., clipnorm=1.)
    ae_model.compile(loss='mean_squared_error', optimizer=clippedAdam)
    print('')
    ae_model.summary()
    
    # reuse decoder to define a model to test generation capacity
    input_point = Input(shape=(latDim,), name='continuous_input')
    discrete_output = DAriA_dcd.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=discrete_output)
    
    first_softmax_evolution = []
    second_softmax_evolution = []
    third_softmax_evolution = []
    for epoch in range(epochs):
        print('epoch:    ', epoch)
        
        print("""
              fit ae
              
              """)
        indices_sentences = next(generator)
        indicess = ae_model.predict(indices_sentences)
        sentences_reconstructed = generator_class.indicesToSentences(indicess)
        ae_model.fit(indices_sentences, indices_sentences, epochs=epochs_in)    
        
        # FIXME: noise in the latent rep
        if epoch%latentTestRate == 0:
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
            indicess, softmaxes = decoder_model.predict(noise)
            sentences_generated = generator_class.indicesToSentences(indicess)
            print('generated sentences')
            print('')
            print(sentences_generated)
            print('')
            print(softmaxes[0][0])
            print('')
            first_softmax_evolution.append(softmaxes[0][0])
            second_softmax_evolution.append(softmaxes[0][1])
            third_softmax_evolution.append(softmaxes[0][2])
             
            
    print(first_softmax_evolution)
    plot_softmax_evolution(first_softmax_evolution, 'first_softmax_evolution')
    print(second_softmax_evolution)
    plot_softmax_evolution(second_softmax_evolution, 'second_softmax_evolution')
    print(third_softmax_evolution)
    plot_softmax_evolution(third_softmax_evolution, 'third_softmax_evolution')
    print('')
    print(generator_class.vocabulary.indicesByTokens)
    print('')
    print(grammar)



def plot_softmax_evolution(softmaxes_list, name='softmaxes'):
    import matplotlib.pylab as plt
    
    f = plt.figure()
    index = range(len(softmaxes_list[0]))
    for softmax in softmaxes_list:
        plt.bar(index, softmax)
        
    
    plt.xlabel('Token')
    plt.ylabel('Probability')    
    plt.title('softmax evolution during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')
        
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
