"""

- [DONE] predict next token
- insert that model inside Simon's implementaiton of AriEL

"""


import sys

import numpy as np
from nltk import CFG
from vAriEL import vAriEL_Encoder_model, vAriEL_Decoder_model, Differential_AriEL
from sentenceGenerators import c2n_generator, next_character_generator
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Reshape, Dense, TimeDistributed, \
                         GaussianNoise, Activation
from keras import optimizers
from keras.callbacks import TensorBoard
from utils import checkDuringTraining, plot_softmax_evolution, make_directories, \
                  TestActiveGaussianNoise, SelfAdjustingGaussianNoise





# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> 'ABC' | 'AAC' | 'BA'
                         """)

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
max_senLen = None
batchSize = 128
# FIXME:
# latDim = 7, embDim = 5 seems to explode with gaussian noise
latDim = 16
embDim = 5

epochs = 2
steps_per_epoch = 1000
epochs_in = 3
latentTestRate = int(epochs_in/10) if not int(epochs_in/10) == 0 else 1






def even_simpler_main(categorical_TF=True):

    # create experiment folder to save the results
    experiment_path = make_directories()
    
    # write everythin in a file
    # sys.stdout = open(experiment_path + 'training.txt', 'w')

    # dataset to be learned
    generator_class = next_character_generator(grammar, batchSize, maxlen=max_senLen, categorical=categorical_TF)
    generator = generator_class.generator()

    vocabSize = generator_class.vocabSize    

    ################################################################################
    # Define the main model, next token prediction
    ################################################################################

    embedding = Embedding(vocabSize, embDim, mask_zero='True')
    lstm = LSTM(128, return_sequences=False)
    
    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocabSize, activation='softmax')(lstm_output)

    optimizer = optimizers.Adam(lr=.001)  # , clipnorm=1.
    
    ae_model = Model(inputs=input_question, outputs=softmax)
    ae_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
    tensorboard = TensorBoard(log_dir='./' + experiment_path + 'log', histogram_freq=latentTestRate,
                              write_graph=True, write_images=True, write_grads=True)
    tensorboard.set_model(ae_model)
    callbacks = [tensorboard]  #  [] # 
    
    ################################################################################
    # Train and test
    ################################################################################
    
    valIndices = next(generator)
    print("""
          fit ae
          
          """)
    before_predictions = ae_model.predict(valIndices[0])
    
    import os
    model_path = 'model_nt.h5'
    if not os.path.isfile(model_path):
        ae_model.fit_generator(generator, 
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs_in, 
                               callbacks=callbacks, 
                               validation_data = valIndices)
        ae_model.save(model_path)
    else:
        from keras.models import load_model
        ae_model = load_model(model_path)
            
    
    prediction = ae_model.predict(valIndices[0])
    

    print('\n\n\n')
    batch_indices = np.argmax(before_predictions, axis=1)
    batch_indices = [[i] for i in batch_indices]
    sentences_reconstructed_before = generator_class.indicesToSentences(batch_indices)
    batch_indices = np.argmax(prediction, axis=1)
    batch_indices = [[i] for i in batch_indices]
    sentences_reconstructed_after = generator_class.indicesToSentences(batch_indices)
    input_sentence = generator_class.indicesToSentences(valIndices[0])
    batch_indices = np.argmax(valIndices[1], axis=1)
    batch_indices = [[i] for i in batch_indices]
    output_sentence = generator_class.indicesToSentences(batch_indices)

    
    
    
    from prettytable import PrettyTable

    table = PrettyTable(['input', 'output', 'reconstruction LM before training', 'reconstruction LM after training'])
    for i,o, b, a in zip(input_sentence, output_sentence, sentences_reconstructed_before, sentences_reconstructed_after):
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
    
    ################################################################################
    # Define the main model, the autoencoder
    ################################################################################
    
    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'both',
                                   embedding = embedding,
                                   rnn = lstm,
                                   startId = generator_class.startId)


    decoder_input = Input(shape=(latDim,), name='decoder_input')
    discrete_output = DAriA_dcd.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)

    
if __name__ == '__main__':
    #main()
    #simpler_main()
    even_simpler_main()
