import os
import sys

from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy

numpy.set_printoptions(threshold=sys.maxsize, suppress=True, precision=3)

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from DifferentiableAriEL.nnets.AriEL import AriEL
from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model

ex = CustomExperiment('2d')


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

    lat_dim = 16  # 2
    vocab_size = np.max(np.max(choices)) + 1
    emb_dim = int(np.sqrt(vocab_size) + 1)

    temp_folder = '../data/tmp/'
    if not os.path.isdir(temp_folder): os.mkdir(temp_folder)

    epochs = 1
    steps_per_epoch = 1e4
    batch_size = 16
    LM_path = temp_folder + 'LM_model.h5'


def choices2NextStep(choices, probabilities):
    ns_choices = []
    ns_probabilities = []
    for sequence, p in zip(choices, probabilities):
        for i in range(len(sequence) - 1):
            ns_choices.append([sequence[:i + 1], [sequence[i + 1]]])
            ns_probabilities.append(p / (len(sequence) - 1))

    return ns_choices, ns_probabilities


@ex.capture
def checkSpaceCoverageDecoder(LM,
                              lat_dim,
                              vocab_size, emb_dim,
                              PAD,
                              tf_RNN=False):
    points = np.random.rand(10000, lat_dim)

    for maxlen in range(1, 6):
        DAriA = AriEL(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            lat_dim=lat_dim,
            maxlen=maxlen,
            output_type='both',
            language_model=LM,
            decoder_type=0,
            PAD=PAD
        )

        input_point = Input(shape=(lat_dim,), name='question')
        point = DAriA.decode(input_point)
        decoder_model = Model(inputs=input_point, outputs=point)

        if not tf_RNN:
            prediction = decoder_model.predict(points)[0].astype(int)
        else:
            prediction = decoder_model.predict(points)[2].astype(int)

        uniques, labels, counts = np.unique(prediction, axis=0, return_inverse=True, return_counts=True)

        t = PrettyTable(['uniques', 'counts'])
        for a in zip(uniques, counts):
            t.add_row([*a])

        print(t)

        fig = plt.figure()
        for i, (seq, c) in enumerate(zip(uniques, counts)):
            if c > 100:
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

        fig.savefig('../data/tmp/decoder_{}.png'.format(maxlen), dpi=fig.dpi)


@ex.capture
def checkSpaceCoverageEncoder(LM, lat_dim, vocab_size, PAD, emb_dim, choices):
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
        for i in range(vocab_size):
            ci.append(last_c + [i])

    for choices in [c0, c1, c2, c3, c4]:
        choices_strings = [''.join([str(n) for n in el]) for el in choices]

        predictions = []
        for sentence in choices:
            maxlen = len(sentence)
            DAriA = AriEL(
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                lat_dim=lat_dim,
                maxlen=maxlen,
                output_type='both',
                language_model=LM,
                encoder_type=1,
                PAD=PAD
            )

            input_questions = Input(shape=(None,), name='question')
            continuous_output = DAriA.encode(input_questions)
            encoder_model = Model(inputs=input_questions, outputs=continuous_output)

            sentence = np.array(sentence)
            prediction = encoder_model.predict(sentence[np.newaxis, :])
            predictions.append(prediction)

        prediction = np.concatenate(predictions, axis=0)

        # print(prediction)

        # uniques, labels = np.unique(padded_q, axis=0, return_inverse=True)

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
def checkReconstruction(LM, lat_dim, vocab_size, PAD, emb_dim, choices):
    maxlen = 100
    batch_size = 20
    encoder_type = 0
    decoder_type = 0
    # vocab_size = 10
    DAriA = AriEL(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        lat_dim=lat_dim,
        maxlen=maxlen,
        output_type='both',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_lat_dim=1e6,
        PAD=PAD
    )

    input_questions = Input(shape=(None,), name='question')
    continuous_output = DAriA.encode(input_questions)
    discrete_output = DAriA.decode(continuous_output)
    reconstruction_model = Model(inputs=input_questions, outputs=discrete_output)

    sentences = np.random.randint(vocab_size, size=(batch_size, maxlen))
    if not decoder_type == 2:
        prediction = reconstruction_model.predict(sentences)[0].astype(int)
    else:
        prediction = reconstruction_model.predict(sentences).astype(int)

    t = PrettyTable(['sentences', 'reconstructions'])
    for a in zip(sentences, prediction):
        t.add_row([*a])

    print(t)
    # print('Are reconstructions perfect? ', np.cumprod(sentences == prediction))

    tot_ = 0
    score = 0
    for s, p in zip(sentences, prediction):
        # print(s)
        # print(p)
        score += np.sum(1 * (s == p))
        tot_ += len(s)
        # print('')
    print(score / tot_)


@ex.capture
def threeSentencesGenerator(choices, probabilities, batch_size=3, vocab_size=5, next_timestep=True):
    if next_timestep:
        choices, probabilities = choices2NextStep(choices, probabilities)

    choices = np.array(choices)

    while True:
        batch = np.random.choice(len(choices), batch_size, p=probabilities)
        batch = choices[batch]

        questions = batch[:, 0]
        padded_q = pad_sequences(questions, padding='pre')
        replies = batch[:, 1]
        padded_r = pad_sequences(replies, padding='pre')
        categorical_r = to_categorical(padded_r, num_classes=vocab_size)
        yield padded_q, categorical_r


@ex.automain
def main_test(vocab_size,
              emb_dim,
              LM_path, epochs, steps_per_epoch,
              _log):
    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough

    generator = threeSentencesGenerator()

    if not os.path.isfile(LM_path):
        LM = predefined_model(vocab_size, emb_dim)
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

    # checkSpaceCoverageDecoder(LM)

    print('\n   Check AriEL Encoder   \n')

    # checkSpaceCoverageEncoder(LM)

    print('\n   Check AriEL AE   \n')

    checkReconstruction(LM)
