"""

- [DONE] predict next token
- insert that model inside Simon's implementaiton of AriEL

"""


from DAriEL import Differentiable_AriEL, predefined_model
from nltk import CFG
from sentenceGenerators import c2n_generator, next_character_generator
from keras import optimizers
from keras.callbacks import TensorBoard
from utils import checkDuringTraining, plot_softmax_evolution, make_directories, \
                  TestActiveGaussianNoise, SelfAdjustingGaussianNoise
from tests import random_sequences_and_points




import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                         Bidirectional, LSTM, Lambda, TimeDistributed, \
                         RepeatVector, Activation, GaussianNoise, Flatten, \
                         Reshape
                         
from keras.utils import to_categorical
import tensorflow as tf
from utils import TestActiveGaussianNoise, SelfAdjustingGaussianNoise





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
max_senLen = 13
batchSize = 128
# FIXME:
# latDim = 7, embDim = 5 seems to explode with gaussian noise
latDim = 16
embDim = 5

epochs = 1
steps_per_epoch = 100
epochs_in = 0
latentTestRate = int(epochs_in/10) if not int(epochs_in/10) == 0 else 1






def main(categorical_TF=True):

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

    ae_model = predefined_model(vocabSize, embDim)

    optimizer = optimizers.Adam(lr=.001)  # , clipnorm=1.    
    ae_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    callbacks = []
    
    
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
    
    DAriEL = Differentiable_AriEL(vocabSize = vocabSize,
                                  embDim = embDim,
                                  latDim = latDim,
                                  max_senLen = max_senLen,
                                  output_type = 'both',
                                  language_model = ae_model,
                                  startId = 0)


    decoder_input = Input(shape=(latDim,), name='decoder_input')
    discrete_output = DAriEL.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)
    
    
    noise = np.random.rand(batchSize, latDim)
    indicess, _ = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)
    
    ################################################################################
    # Subjective evaluation of the quality of the sentences
    ################################################################################
    
    from prettytable import PrettyTable

    table = PrettyTable(['input', 'output', 'reconstruction LM before training', 'reconstruction LM after training', 'DAriA generated with that LM'])
    for i,o, b, a, g in zip(input_sentence, output_sentence, sentences_reconstructed_before, sentences_reconstructed_after, sentences_generated):
        table.add_row([i, o, b, a, g])
    for column in table.field_names:        
        table.align[column] = "l"
    print(table)
    print('')
    print('number unique generated sentences:  %s / %s (it should be only 3 / %s)'%(len(set(sentences_generated)), batchSize, batchSize))
    print('')
    print(generator_class.vocabulary.indicesByTokens)
    print('')
    print(grammar)
    print(ae_model.predict([0]))

    

    
if __name__ == '__main__':
    main()
    
    
    
    
    