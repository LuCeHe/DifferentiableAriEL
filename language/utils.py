# Copyright (c) 2018, 
#
# authors: Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os, time
import numpy as np
from time import strftime, localtime

import tensorflow.keras.backend as K
# from tensorflow.keras.legacy import interfaces
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.engine import InputSpec
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def make_directories(time_string=None):
    
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
    plt.title('softmax evoluti\on during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')
    
    
def checkDuringTraining(generator_class, indices_sentences, encoder_model, decoder_model, batchSize, latDim):
    
    # original sentences
    sentences = generator_class.indicesToSentences(indices_sentences)

    # reconstructed sentences
    point = encoder_model.predict(indices_sentences)
    indices_reconstructed, _ = decoder_model.predict(point)    

    sentences_reconstructed = generator_class.indicesToSentences(indices_reconstructed)

    # generated sentences
    noise = np.random.rand(batchSize, latDim)
    indicess, softmaxes = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)

    from prettytable import PrettyTable

    table = PrettyTable(['original', 'reconstructed', 'generated'])
    for b, a, g in zip(sentences, sentences_reconstructed, sentences_generated):
        table.add_row([b, a, g])
    for column in table.field_names:        
        table.align[column] = "l"
    print(table)
    
    print('')
    print('number unique generated sentences:   ', len(set(sentences_generated)))
    print('')
    print(softmaxes[0][0])
    print('')
    
    return softmaxes


def random_sequences_and_points(batchSize=3,
                                latDim=4,
                                max_senLen=6,
                                repeated=False,
                                vocabSize=2,
                                hyperplane=False):
    
    if not repeated:
        questions = []
        points = np.random.rand(batchSize, latDim)
        for _ in range(batchSize):
            sentence_length = max_senLen  # np.random.choice(max_senLen)
            randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
            # EOS = (vocabSize+1)*np.ones(1)
            # randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, latDim)
        sentence_length = max_senLen  # np.random.choice(max_senLen)
        question = np.random.choice(vocabSize, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats=[batchSize], axis=0)
        questions = np.repeat(question, repeats=[batchSize], axis=0)
        
    padded_questions = pad_sequences(questions)
    
    if hyperplane:
        point_1 = np.random.rand(1, 1)
        point_1 = np.repeat(point_1, repeats=[batchSize], axis=0)
        point_2 = np.random.rand(batchSize, latDim - 1)
        points = np.concatenate([point_1, point_2], axis=1)
        
    return padded_questions, points


    
def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string