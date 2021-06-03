# learn language in the AE


"""

- [DONE] instead of feeding embedding/lstm/TimeDistributed from outside, just feed anymodel discrete -> softmax
- [DONE] so I can plug in more complex models
- [DONE] make it next token prediction

"""

import numpy as np
from DAriEL import DAriEL_Encoder_model, DAriEL_Decoder_model, Differentiable_AriEL, \
    predefined_model

from nltk import CFG
from sentenceGenerators import c2n_generator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Reshape, Dense, TimeDistributed, \
    GaussianNoise, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from utils import checkDuringTraining, plot_softmax_evolution, make_directories

from tensorflow.keras.utils import to_categorical
from DifferentiableAriEL.tests import random_sequences_and_points

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

vocab_size = 3  # this value is going to be overwriter after the sentences generator
maxlen = 6
batch_size = 128
# FIXME:
# lat_dim = 7, emb_dim = 5 seems to explode with gaussian noise
lat_dim = 16
emb_dim = 5

epochs = 2
steps_per_epoch = 1000
epochs_in = 3
latentTestRate = int(epochs_in / 10) if not int(epochs_in / 10) == 0 else 1


def main(categorical_TF=True):
    # create experiment folder to save the results
    experiment_path = make_directories()

    # write everythin in a file
    # sys.stdout = open(experiment_path + 'training.txt', 'w')

    # dataset to be learned
    generator_class = c2n_generator(grammar, batch_size, maxlen=maxlen, categorical=categorical_TF)
    generator = generator_class.generator()

    vocab_size = generator_class.vocab_size

    ################################################################################
    # Define the main model, the autoencoder
    ################################################################################

    DAriA_dcd = DAriEL(vocab_size=vocab_size,
                       emb_dim=emb_dim,
                       lat_dim=lat_dim,
                       maxlen=maxlen,
                       startId=generator_class.startId,
                       output_type='both')

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    # continuous_latent_space = TestActiveGaussianNoise(stddev=.08)(continuous_latent_space)
    # continuous_latent_space = SelfAdjustingGaussianNoise()(continuous_latent_space)

    discrete_output = DAriA_dcd.decode(continuous_latent_space)

    optimizer = optimizers.Adam(lr=.01)  # , clipnorm=1.
    if categorical_TF:
        ae_model = Model(inputs=input_question, outputs=discrete_output[1])
        # ae_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # approximation to Wasserstain distance
        ae_model.compile(loss='mean_absolute_error', optimizer=optimizer)
    else:
        ae_model = Model(inputs=input_question, outputs=discrete_output[0])
        ae_model.compile(loss='mean_absolute_error', optimizer=optimizer)

    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,
                              write_graph=True, write_images=True, write_grads=True)
    tensorboard.set_model(ae_model)
    callbacks = [tensorboard]  # [] #

    ################################################################################
    # reuse encoder and decoder to check behavior
    ################################################################################

    # reuse encoder to define a model to test encoding capacity
    encoder_input = Input(shape=(None,), name='encoder_input')
    continuous_latent_space = DAriA_dcd.encode(encoder_input)
    encoder_model = Model(inputs=encoder_input, outputs=continuous_latent_space)

    # reuse decoder to define a model to test generation capacity
    decoder_input = Input(shape=(lat_dim,), name='decoder_input')
    discrete_output = DAriA_dcd.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)

    first_softmax_evolution = []
    second_softmax_evolution = []
    third_softmax_evolution = []

    ################################################################################
    # Train and test
    ################################################################################

    valIndices = next(generator)
    for epoch in range(epochs):
        print('epoch:    ', epoch)

        print("""
              fit ae
              
              """)
        indices_sentences = next(generator)
        # predictions = ae_model.predict(indices_sentences[0])
        # print(predictions)
        ae_model.fit_generator(generator,
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs_in,
                               callbacks=callbacks,
                               validation_data=valIndices)

        # FIXME: noise in the latent rep
        if epoch % latentTestRate == 0:
            softmaxes = checkDuringTraining(generator_class, indices_sentences[0], encoder_model, decoder_model,
                                            batch_size, lat_dim)

            first_softmax_evolution.append(softmaxes[0][0])
            second_softmax_evolution.append(softmaxes[0][1])
            third_softmax_evolution.append(softmaxes[0][2])

    # softmaxes = checkDuringTraining(generator_class, indices_sentences[0], encoder_model, decoder_model, batch_size, lat_dim)

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


def simpler_main(categorical_TF=True):
    # create experiment folder to save the results
    experiment_path = make_directories()

    # write everythin in a file
    # sys.stdout = open(experiment_path + 'training.txt', 'w')

    # dataset to be learned
    generator_class = c2n_generator(grammar, batch_size, maxlen=maxlen, categorical=categorical_TF)
    generator = generator_class.generator()

    vocab_size = generator_class.vocab_size

    ################################################################################
    # Define the main model, the autoencoder
    ################################################################################

    embedding = Embedding(vocab_size, emb_dim)
    lstm = LSTM(vocab_size, return_sequences=True)

    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = TimeDistributed(Activation('softmax'))(lstm_output)

    optimizer = optimizers.Adam(lr=.01)  # , clipnorm=1.

    ae_model = Model(inputs=input_question, outputs=softmax)
    ae_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,
                              write_graph=True, write_images=True, write_grads=True)
    tensorboard.set_model(ae_model)
    callbacks = [tensorboard]  # [] #

    ################################################################################
    # Train and test
    ################################################################################

    valIndices = next(generator)
    print("""
          fit ae
          
          """)
    before_predictions = ae_model.predict(valIndices[0])

    ae_model.fit_generator(generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=epochs_in,
                           callbacks=callbacks,
                           validation_data=valIndices)

    prediction = ae_model.predict(valIndices[0])

    print('\n\n\n')
    batch_indices = np.argmax(before_predictions, axis=2)
    sentences_reconstructed_before = generator_class.indicesToSentences(batch_indices)
    batch_indices = np.argmax(prediction, axis=2)
    sentences_reconstructed_after = generator_class.indicesToSentences(batch_indices)
    input_sentence = generator_class.indicesToSentences(valIndices[0])
    output_sentence = generator_class.indicesToSentences(np.argmax(valIndices[1], axis=2))

    ################################################################################
    # Define the main model, the autoencoder
    ################################################################################

    DAriA_dcd = Differential_AriEL(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   lat_dim=lat_dim,
                                   maxlen=maxlen,
                                   output_type='both',
                                   embedding=embedding,
                                   rnn=lstm,
                                   startId=generator_class.startId)

    decoder_input = Input(shape=(lat_dim,), name='decoder_input')
    discrete_output = DAriA_dcd.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)

    noise = np.random.rand(batch_size, lat_dim)
    indicess, _ = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)

    from prettytable import PrettyTable

    table = PrettyTable(['input', 'output', 'reconstruction LM before training', 'reconstruction LM after training',
                         'DAriA generated with that LM'])
    for i, o, b, a, g in zip(input_sentence, output_sentence, sentences_reconstructed_before,
                             sentences_reconstructed_after, sentences_generated):
        table.add_row([i, o, b, a, g])
    for column in table.field_names:
        table.align[column] = "l"
    print(table)
    print('')
    print('number unique generated sentences:  %s / %s (it should be only 3 / %s)' % (
    len(set(sentences_generated)), batch_size, batch_size))
    print('')
    print(generator_class.vocabulary.indicesByTokens)
    print('')
    print(grammar)
    print(ae_model.predict([0]))


def even_simpler_main(categorical_TF=True):
    # create experiment folder to save the results
    experiment_path = make_directories()

    # write everythin in a file
    # sys.stdout = open(experiment_path + 'training.txt', 'w')

    # dataset to be learned
    generator_class = c2n_generator(grammar, batch_size, maxlen=maxlen, categorical=categorical_TF)
    generator = generator_class.generator()

    vocab_size = generator_class.vocab_size

    ################################################################################
    # Define the main model, the autoencoder
    ################################################################################

    embedding = Embedding(vocab_size, emb_dim)
    lstm = LSTM(128, return_sequences=True)

    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm_output)

    optimizer = optimizers.Adam(lr=.001)  # , clipnorm=1.

    ae_model = Model(inputs=input_question, outputs=softmax)
    ae_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,
                              write_graph=True, write_images=True, write_grads=True)
    tensorboard.set_model(ae_model)
    callbacks = [tensorboard]  # [] #

    ################################################################################
    # Train and test
    ################################################################################

    valIndices = next(generator)
    print("""
          fit ae
          
          """)
    before_predictions = ae_model.predict(valIndices[0])

    import os
    if not os.path.isfile('model.h5'):
        ae_model.fit_generator(generator,
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs_in,
                               callbacks=callbacks,
                               validation_data=valIndices)
        ae_model.save('model.h5')
    else:
        from keras.models import load_model
        ae_model = load_model('model.h5')

    prediction = ae_model.predict(valIndices[0])

    print('\n\n\n')
    batch_indices = np.argmax(before_predictions, axis=2)
    sentences_reconstructed_before = generator_class.indicesToSentences(batch_indices)
    batch_indices = np.argmax(prediction, axis=2)
    sentences_reconstructed_after = generator_class.indicesToSentences(batch_indices)
    input_sentence = generator_class.indicesToSentences(valIndices[0])
    output_sentence = generator_class.indicesToSentences(np.argmax(valIndices[1], axis=2))

    from prettytable import PrettyTable

    table = PrettyTable(['input', 'output', 'reconstruction LM before training', 'reconstruction LM after training'])
    for i, o, b, a in zip(input_sentence, output_sentence, sentences_reconstructed_before,
                          sentences_reconstructed_after):
        table.add_row([i, o, b, a])
    for column in table.field_names:
        table.align[column] = "l"
    print(table)
    print('')
    print('')
    print(generator_class.vocabulary.indicesByTokens)
    print('')
    print(grammar)
    print(ae_model.predict([0]))


def test_DAriEL_model_from_outside_v2():
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    answers = to_categorical(questions[:, 1], vocab_size)
    print(answers)
    LM = predefined_model(vocab_size, emb_dim)
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])

    LM.fit(questions, answers, epochs=10)

    DAriEL = Differentiable_AriEL(vocab_size=vocab_size,
                                  emb_dim=emb_dim,
                                  lat_dim=lat_dim,
                                  maxlen=maxlen,
                                  output_type='both',
                                  language_model=LM,
                                  startId=0)

    decoder_input = Input(shape=(lat_dim,), name='decoder_input')
    discrete_output = DAriEL.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)

    noise = np.random.rand(batch_size, lat_dim)
    indicess, _ = decoder_model.predict(noise)

    prediction = decoder_model.predict(points)
    print()
    print(indicess)


if __name__ == '__main__':
    # main()
    # simpler_main()
    # even_simpler_main()
    test_DAriEL_model_from_outside_v2()
