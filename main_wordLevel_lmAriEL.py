"""

- [DONE] predict next token
- insert that model inside Simon's implementaiton of AriEL

"""

import os


from DAriEL import Differentiable_AriEL, predefined_model
import nltk
from nltk import CFG
from sentenceGenerators import c2n_generator, next_character_generator, next_word_generator, w2n_generator
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

from prettytable import PrettyTable




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


grammar = nltk.data.load('data/word_grammar.cfg')
vocab_size = 831  #36  # this value is going to be overwriter after the sentences generator
max_senLen = 14
batch_size = 128
# FIXME:
# lat_dim = 7, emb_dim = 5 seems to explode with gaussian noise
lat_dim = 16
emb_dim = 5

epochs = 1
steps_per_epoch = 10
epochs_in = 3
latentTestRate = int(epochs_in/10) if not int(epochs_in/10) == 0 else 1

model_path = 'data/model_nw_full_grammar.h5'




def checkAutoencoder(ae_model, valIndices):
    
    categorical_TF=True
    generator_class = w2n_generator(grammar, batch_size, maxlen=max_senLen, categorical=categorical_TF)
    generator = generator_class.generator()
    valIndices = next(generator)
    data = np.argmax(valIndices[1], axis=2)  # valIndices[0]    #



    DAriEL = Differentiable_AriEL(vocab_size = vocab_size,
                                  emb_dim = emb_dim,
                                  lat_dim = lat_dim,
                                  max_senLen = max_senLen,
                                  output_type = 'tokens',
                                  language_model = ae_model,
                                  startId = 0)


    encoder_input = Input(shape=(None,), name='encoder_input')  
    continuous_output = DAriEL.encode(encoder_input)
    discrete_reconstruction = DAriEL.decode(continuous_output)
    DAriEL_model = Model(inputs=encoder_input, outputs=discrete_reconstruction)
    print(DAriEL_model.summary())

    
    
    
    prediction = DAriEL_model.predict(data)
    
    nbSuccess = 0
    for d in data:
        d_expanded = np.expand_dims(d, axis=0)
        p = DAriEL_model.predict(d_expanded)[0]
        print(p)
        print(d)
        d_questionMark = list(d).index(4)
        
        isClose = np.allclose(p[:d_questionMark+1],d[:d_questionMark+1])
        #isClose = np.allclose(p,d)
        print('p=d?   ', isClose)
        if isClose:
            nbSuccess+=1
        print('')
    print('DAriEL successes?   %s/%s = %s%%'%(nbSuccess, len(data), int(nbSuccess/len(data)*10000)/100))
    print('DAriEL reconstruction is perfect?   ', np.allclose(prediction,data))



def checkEncoder(ae_model, valIndices):
    DAriEL = Differentiable_AriEL(vocab_size = vocab_size,
                                  emb_dim = emb_dim,
                                  lat_dim = lat_dim,
                                  max_senLen = max_senLen,
                                  output_type = 'both',
                                  language_model = ae_model,
                                  startId = 0)


    encoder_input = Input(shape=(None,), name='decoder_input')  
    continuous_output = DAriEL.encode(encoder_input)
    encoder_model = Model(inputs=encoder_input, outputs=continuous_output)
    print(encoder_model.summary())
    prediction = encoder_model.predict(valIndices[0])
    print(prediction)
    
def checkDecoder(ae_model, generator_class, input_sentence, output_sentence, sentences_reconstructed_before, sentences_reconstructed_after):
    DAriEL = Differentiable_AriEL(vocab_size = vocab_size,
                                  emb_dim = emb_dim,
                                  lat_dim = lat_dim,
                                  max_senLen = max_senLen,
                                  output_type = 'both',
                                  language_model = ae_model,
                                  startId = 0)


    decoder_input = Input(shape=(lat_dim,), name='decoder_input')
    discrete_output = DAriEL.decode(decoder_input)
    decoder_model = Model(inputs=decoder_input, outputs=discrete_output)
    
    
    noise = np.random.rand(batch_size, lat_dim)
    indicess, _ = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)
    
    ################################################################################
    # Subjective evaluation of the quality of the sentences
    ################################################################################
    

    table = PrettyTable(['input', 'output', 'reconstruction LM before training', 'reconstruction LM after training', 'DAriA generated with that LM'])
    for i,o, b, a, g in zip(input_sentence, output_sentence, sentences_reconstructed_before, sentences_reconstructed_after, sentences_generated):
        table.add_row([i, o, b, a, g])
    for column in table.field_names:        
        table.align[column] = "l"
    print(table)
    print('')
    print('number unique generated sentences:  %s / %s (it should be only 3 / %s)'%(len(set(sentences_generated)), batch_size, batch_size))
    print('')
    print(grammar)
    print(ae_model.predict([0]))
    
    
    for p in indicess:
        print(p)
        


def main(categorical_TF=True):

    # create experiment folder to save the results
    experiment_path = make_directories()
    
    # write everythin in a file
    # sys.stdout = open(experiment_path + 'training.txt', 'w')

    # dataset to be learned
    generator_class = next_word_generator(grammar, batch_size, maxlen=max_senLen, categorical=categorical_TF)
    generator = generator_class.generator()

    vocab_size = generator_class.vocab_size

    ################################################################################
    # Define the main model, next token prediction
    ################################################################################

    ae_model = predefined_model(vocab_size, emb_dim)
    print(ae_model.summary())

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

    
    """
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
    """
    
    
    ################################################################################
    # Define the main model, the autoencoder, and check Decoder
    ################################################################################

    #checkDecoder(ae_model, generator_class, input_sentence, output_sentence, sentences_reconstructed_before, sentences_reconstructed_after)

    ################################################################################
    # Check if encoder works: DONE!
    ################################################################################
    
    #checkEncoder(ae_model, valIndices)    
    
    ################################################################################
    # Check if autoencoder works
    ################################################################################
    
    checkAutoencoder(ae_model, valIndices)

    print(generator_class.vocabulary.indicesByTokens)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    