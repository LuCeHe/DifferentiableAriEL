#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:03:21 2019

@author: perfect
"""
import os
import numpy as np
from time import strftime, localtime


def make_directories(time_string = None):
    
    experiments_folder = "experiments"
    if not os.path.isdir(experiments_folder):
        os.mkdir(experiments_folder)    
        
    if time_string == None:
        time_string = strftime("%Y-%m-%d-at-%H:%M:%S", localtime())
    
    experiment_folder = experiments_folder + '/experiment-' + time_string + '/'
    
    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)          
        
    # create folder to save new models trained
    model_folder = experiment_folder + '/model/'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)   

    # create folder to save TensorBoard
    log_folder = experiment_folder + '/log/'
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)   
                
    return experiment_folder



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
        
    
    
    
    
def checkDuringTraining(generator_class, indices_sentences, ae_model, decoder_model, batchSize, latDim):
    
    
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
    
    return softmaxes

    