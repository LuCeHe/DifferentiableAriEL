import sys, os

import numpy
numpy.set_printoptions(threshold=sys.maxsize, suppress=True, precision=3)

import time 
import matplotlib
from tensorflow.keras.callbacks import TensorBoard
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                         Bidirectional, LSTM, Lambda, TimeDistributed, \
                         RepeatVector, Activation, GaussianNoise, Flatten, \
                         Reshape
 
from tensorflow.keras.models import load_model                        
from tensorflow.keras.utils import to_categorical

from nnets.DAriEL import DAriEL_Encoder_model, DAriEL_Decoder_model, Differentiable_AriEL, \
    predefined_model, DAriEL_Encoder_Layer, DAriEL_Decoder_Layer

import os, sys, copy, logging
import pathlib
from time import strftime, localtime

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


class CustomFileStorageObserver(FileStorageObserver):

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # create your wanted log dir
            time_string = strftime("%Y-%m-%d-at-%H:%M:%S", localtime())
            timestamp = "experiment-{}________".format(time_string)
            options = '_'.join(meta_info['options']['UPDATE'])
            run_id = timestamp + options
            
            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)
            
            # and again create the basedir
            pathlib.Path(self.basedir).mkdir(exist_ok=True, parents=True)
        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)


ex = Experiment('LSNN_AE')
# ex.observers.append(FileStorageObserver.create("experiments"))
ex.observers.append(CustomFileStorageObserver.create("experiments"))

# ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds

# set up a custom logger
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

# attach it to the experiment
ex.logger = logger


@ex.config
def cfg():
    
    PAD = 0
    START = 1
    END = 2
    TOKEN_1 = 3
    TOKEN_2 = 4

    choices = [[START, TOKEN_1, TOKEN_2, TOKEN_2, END],
               [START, TOKEN_2, END],
               [START, TOKEN_2, TOKEN_1, END],
               [START, TOKEN_2, TOKEN_1, TOKEN_1, END]
               ]
    probabilities = [.25, .25, .25, .25]
    
    latDim = 2
    vocabSize = np.max(np.max(choices)) + 1
    embDim = int(np.sqrt(vocabSize) + 1)

    temp_folder = '../data/tmp/'
    if not os.path.isdir(temp_folder): os.mkdir(temp_folder)
    
    epochs = 1
    steps_per_epoch = 1e4
    batchSize = 16
    LM_path = temp_folder + 'LM_model.h5'


def choices2NextStep(choices, probabilities):
    ns_choices = []
    ns_probabilities = []
    for sequence, p in zip(choices, probabilities):
        for i in range(len(sequence) - 1):
            ns_choices.append([sequence[:i + 1], [sequence[i + 1]]])
            ns_probabilities.append(p / (len(sequence) - 1))
    
    return ns_choices, ns_probabilities


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


@ex.capture
def checkSpaceCoverageDecoder(LM,
                              latDim,
                              vocabSize, embDim,
                              PAD,
                              tf_RNN=True):

    _, points = random_sequences_and_points(batchSize=10000, latDim=latDim)
    _, points = random_sequences_and_points(batchSize=10, latDim=latDim)
    # _, points = random_sequences_and_points(batchSize=10000, latDim=latDim, hyperplane=True)

    for max_senLen in range(3, 6):
        DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                     embDim=embDim,
                                     latDim=latDim,
                                     max_senLen=max_senLen,
                                     output_type='both',
                                     language_model=LM,
                                     tf_RNN=tf_RNN,
                                     PAD=PAD)
        
        input_point = Input(shape=(latDim,), name='question')
        point = DAriA.decode(input_point)
        decoder_model = Model(inputs=input_point, outputs=point)
        
        if not tf_RNN:
            prediction = decoder_model.predict(points)[0].astype(int)
        else:
            prediction = decoder_model.predict(points)[1].astype(int)
        
        uniques, labels, counts = np.unique(prediction, axis=0, return_inverse=True, return_counts=True)
        
        t = PrettyTable(['uniques', 'counts'])
        for a in zip(uniques, counts):
            t.add_row([*a])
        
        print(t)
        
        fig = plt.figure()    
        for i, (seq, c) in enumerate(zip(uniques, counts)):
            if c > 20:
                label = labels == i
                seq_string = ''.join([str(n) for n in seq]) 
                
                plt.scatter(points[label, 0], points[label, 1],
                            label=seq_string,
                            s=50, cmap='gist_ncar');
    
        axes = plt.gca()
        
        plt.legend()
        eps = .1
        axes.set_xlim([0 - eps, 1 + eps])
        axes.set_ylim([0 - eps, 1 + eps])
        plt.show()
        
        fig.savefig('../data/tmp/decoder_{}.png'.format(max_senLen), dpi=fig.dpi)


@ex.capture
def checkSpaceCoverageEncoder(LM, latDim, vocabSize, PAD, embDim, choices):
    
    c0 = choices
    
    c1 = []
    for c in [c[:2] for c in choices]:
        if c not in c1:
            c1.append(c)
            
    c2follow = choices[-1]

    
    c2 = [c2follow[:2]]
    c3 = [c2follow[:2], c2follow[:3]]
    c4 = [c2follow[:2], c2follow[:3], c2follow[:4]]
    
    cs = [c2, c3, c4]
    for ci in cs:
        last_c = ci[-1]
        for i in range(vocabSize):
            ci.append(last_c + [i])
    

    
    for choices in [c0, c1, c2, c3, c4]:
        choices_strings = [''.join([str(n) for n in el]) for el in choices]
        
        predictions = []
        for sentence in choices:
            max_senLen = len(sentence)
            DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                         embDim=embDim,
                                         latDim=latDim,
                                         max_senLen=max_senLen,
                                         output_type='both',
                                         language_model=LM,
                                         PAD=PAD)
            
            input_questions = Input(shape=(None,), name='question')    
            continuous_output = DAriA.encode(input_questions)
            encoder_model = Model(inputs=input_questions, outputs=continuous_output)
            
            sentence = np.array(sentence)
            prediction = encoder_model.predict(sentence[np.newaxis, :])
            predictions.append(prediction)
        
        prediction = np.concatenate(predictions, axis=0)
    
        #print(prediction)
        
        #uniques, labels = np.unique(padded_q, axis=0, return_inverse=True)
        
        for sample, string in zip(prediction, choices_strings):
            plt.scatter(sample[0], sample[1],
                        s=50, cmap='nipy_spectral', label=string)
        axes = plt.gca()
        
        plt.legend()
        eps = .1
        axes.set_xlim([0 - eps, 1 + eps])
        axes.set_ylim([0 - eps, 1 + eps])
        plt.show()

@ex.capture
def threeSentencesGenerator(choices, probabilities, batchSize=3, vocabSize=5, next_timestep=True):
    
    if next_timestep:
        choices, probabilities = choices2NextStep(choices, probabilities)

    choices = np.array(choices)
    
    while True:
        batch = np.random.choice(len(choices), batchSize, p=probabilities)       
        batch = choices[batch]

        questions = batch[:, 0]
        padded_q = pad_sequences(questions, padding='pre')
        replies = batch[:, 1]
        padded_r = pad_sequences(replies, padding='pre')
        categorical_r = to_categorical(padded_r, num_classes=vocabSize)  
        yield padded_q, categorical_r


@ex.automain
def test_2d_visualization_trainOutside(vocabSize,
                                       embDim,
                                       latDim,
                                       choices,
                                       probabilities,
                                       START, END, PAD,
                                       LM_path, epochs, steps_per_epoch,
                                       _log):
    
    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough
    
    generator = threeSentencesGenerator()
    
    if not os.path.isfile(LM_path):
        LM = predefined_model(vocabSize, embDim)
        LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['categorical_accuracy'])
        LM.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)  
        LM.save(LM_path)
    else:
        LM = load_model(LM_path)
            
    print('\n   Check LM   \n')
    
    batch = next(generator)
    output = LM.predict(batch[0])
    
    t = PrettyTable(['batch', 'pred'])
    for a in zip(batch[0], output):
        t.add_row([*a])
    
    print(t)

    print('\n   Check AriEL Decoder   \n')

    checkSpaceCoverageDecoder(LM)

    print('\n   Check AriEL Encoder   \n')
    
    #checkSpaceCoverageEncoder(LM)

