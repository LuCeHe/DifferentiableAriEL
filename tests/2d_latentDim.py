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


def random_sequences_and_points(batchSize=3, latDim=4, max_senLen=6, repeated=False, vocabSize=vocabSize):
    
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

    return padded_questions, points


def checkSpaceCoverageDecoder(decoder_model, latDim, max_senLen):
    _, points = random_sequences_and_points(batchSize=10000, latDim=latDim, max_senLen=max_senLen)
    prediction = decoder_model.predict(points)
    prediction = np.argmax(prediction, axis=1)
    
    uniques, labels, counts = np.unique(prediction, axis=0, return_inverse=True, return_counts=True)
    
    t = PrettyTable(['uniques', 'counts'])
    for a in zip(uniques, counts):
        t.add_row([*a])
    
    print(t)

    plt.scatter(points[:, 0], points[:, 1], c=labels,
                s=50, cmap='gist_ncar');
    plt.show()


def checkSpaceCoverageEncoder(encoder_model, latDim, max_senLen):
    
    choices = [[1, 2, 2],
               [2],
               [2, 1],
               [2, 1, 1]
               ]
    
    padded_q = pad_sequences(choices, padding='post')    
    print(padded_q)
    prediction = encoder_model.predict(padded_q)
    
    print(prediction)
    
    uniques, labels = np.unique(padded_q, axis=0, return_inverse=True)

    plt.scatter(prediction[:, 0], prediction[:, 1], c=labels,
                s=50, cmap='gist_ncar', label=choices);
    plt.show()


def threeSentencesGenerator(batchSize=3, next_timestep=True):
    
    if not next_timestep:
        choices = [[3, 1, 2, 2],
                   [3, 2],
                   [3, 2, 1],
                   [3, 2, 1, 1]
                   ]
        probabilities = [.25, .25, .25, .25]
    else:
        choices = [[[3, 1, 2], [2]],
               [[3, 1], [2]],
               [[3], [1]],
               [[3], [2]],
               [[3, 2], [1]],
               [[3], [2]],
               [[3, 2, 1], [1]],
               [[3, 2], [1]],
               [[3], [2]],
               ]
        probabilities = [.5 / 3,
                         .5 / 3,
                         .5 / 3,
                         .3,
                         .1 / 2,
                         .1 / 2,
                         .1 / 3,
                         .1 / 3,
                         .1 / 3
                         ]

    choices = np.array(choices)
    
    while True:
        batch = np.random.choice(len(choices), batchSize, p=probabilities)       
        batch = choices[batch]

        questions = batch[:, 0]
        padded_q = pad_sequences(questions, padding='pre')
        replies = batch[:, 1]
        padded_r = pad_sequences(replies, padding='pre')
        categorical_r = to_categorical(padded_r, num_classes=4)  
        yield padded_q, categorical_r


def test_2d_visualization_trainOutside():
    
    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough
    
    max_senLen = 4
    vocabSize = 4
    embDim = int(np.sqrt(vocabSize) + 1)
    latDim = 2
    epochs = 1
    steps_per_epoch = 1e4
    startId = 3
    batchSize = 16
    LM_path = 'data/LM_model.h5'
    
    generator = threeSentencesGenerator(batchSize)
    
    if not os.path.isfile(LM_path):
        LM = predefined_model(vocabSize, embDim)
        LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['categorical_accuracy'])
        LM.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)  
        LM.save(LM_path)
    else:
        LM = load_model(LM_path)
    
    DAriA = Differentiable_AriEL(vocabSize=vocabSize,
                                 embDim=embDim,
                                 latDim=latDim,
                                 max_senLen=max_senLen,
                                 output_type='both',
                                 language_model=LM,
                                 startId=startId)
            
    print('\n   Check LM   \n')
    
    batch = next(generator)
    output = LM.predict(batch[0])
    
    t = PrettyTable(['batch', 'pred'])
    for a in zip(batch[0], output):
        t.add_row([*a])
    
    print(t)
    """

    print('\n   Check AriEL Decoder   \n')

    input_point = Input(shape=(latDim,), name='question')    
    point = DAriA.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=point[1])

    checkSpaceCoverageDecoder(decoder_model, latDim, max_senLen)
    
    """
    
    print('\n   Check AriEL Encoder   \n')

    input_questions = Input(shape=(None,), name='question')    
    continuous_output = DAriA.encode(input_questions)
    encoder_model = Model(inputs=input_questions, outputs=continuous_output)
    
    checkSpaceCoverageEncoder(encoder_model, latDim, max_senLen)

    
if __name__ == '__main__':
    test_2d_visualization_trainOutside()
    
