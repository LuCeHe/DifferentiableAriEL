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
from keras.layers import Input, LSTM, Embedding, Reshape, Dense, TimeDistributed, \
                         GaussianNoise
from keras import optimizers
from keras.callbacks import TensorBoard
from utils import checkDuringTraining, plot_softmax_evolution, make_directories



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
max_senLen = 4 #24
batchSize = 128
# FIXME:
# latDim = 7, embDim = 5 seems to explode with gaussian noise
latDim = 5
embDim = 2
epochs = 1
epochs_in = 10
latentTestRate = int(epochs_in/10)




    
def main():

    # create experiment folder to save the results
    experiment_path = make_directories()
    
    # dataset to be learned<
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
    #continuous_latent_space = GaussianNoise(stddev=0.2)(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    ae_model = Model(inputs=input_question, outputs=discrete_output[0])   # + [continuous_latent_space])        
    
    
    clippedAdam = optimizers.Adam(lr=.02, clipnorm=1.)
    ae_model.compile(loss='mean_absolute_error', optimizer=clippedAdam)
    print('')
    ae_model.summary()
    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,  
                              write_graph=True, write_images=True, write_grads=True)
    tensorboard.set_model(ae_model)
    callbacks = [] # [tensorboard]
    
    # reuse decoder to define a model to test generation capacity
    input_point = Input(shape=(latDim,), name='continuous_input')
    discrete_output = DAriA_dcd.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=discrete_output)
    
    first_softmax_evolution = []
    second_softmax_evolution = []
    third_softmax_evolution = []
    
    
    valIndices = next(generator)
    for epoch in range(epochs):
        print('epoch:    ', epoch)
        
        print("""
              fit ae
              
              """)
        indices_sentences = next(generator)
        ae_model.fit(indices_sentences, indices_sentences, epochs=epochs_in, 
                     callbacks=callbacks, validation_data = (valIndices, valIndices))    
        
        # FIXME: noise in the latent rep
        if epoch%latentTestRate == 0:
            softmaxes = checkDuringTraining(generator_class, indices_sentences, ae_model, decoder_model, batchSize, latDim)

            first_softmax_evolution.append(softmaxes[0][0])
            second_softmax_evolution.append(softmaxes[0][1])
            third_softmax_evolution.append(softmaxes[0][2])
             
            
    print(first_softmax_evolution)
    plot_softmax_evolution(first_softmax_evolution, experiment_path + 'first_softmax_evolution')
    print(second_softmax_evolution)
    plot_softmax_evolution(second_softmax_evolution, experiment_path + 'second_softmax_evolution')
    print(third_softmax_evolution)
    plot_softmax_evolution(third_softmax_evolution, experiment_path + 'third_softmax_evolution')
    print('')
    print(generator_class.vocabulary.indicesByTokens)
    print('')
    print(grammar)


    
def main_simple():

    # create experiment folder to save the results
    experiment_path = make_directories()
    
    # dataset to be learned
    generator_class = c2n_generator(grammar, batchSize, maxlen=max_senLen)
    generator = generator_class.generator()

    vocabSize = generator_class.vocabSize    
    
    print("""
          Test Auto-Encoder on Grammar
          
          """)        


    input_question = Input(shape=(max_senLen,), name='discrete_sequence', dtype='int32')
    embed = Embedding(vocabSize, embDim)(input_question)
    rnn_output = LSTM(vocabSize, return_sequences=True)(embed)
    output = TimeDistributed(Dense(1, activation='sigmoid'))(rnn_output)
    output = Reshape((max_senLen,))(output)
    
    ae_model = Model(inputs=input_question, outputs=output)   # + [continuous_latent_space])        
    
    
    clippedAdam = optimizers.Adam(lr=2., clipnorm=1.)
    ae_model.compile(loss='binary_crossentropy', optimizer=clippedAdam)
    print('')
    ae_model.summary()
    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,  
                              write_graph=False, write_images=False, write_grads=True)
    tensorboard.set_model(ae_model)
    
    valIndices = next(generator)
    print(valIndices.shape)
    for epoch in range(epochs):
        print('epoch:    ', epoch)
        
        indices_sentences = next(generator) 
        ae_model.fit(indices_sentences, indices_sentences, epochs=epochs_in, 
                     validation_data = (valIndices, valIndices), callbacks=[tensorboard], verbose=1)   


    
    
if __name__ == '__main__':
    main()
