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



import numpy as np
from DAriEL import DAriEL_Encoder_model, DAriEL_Decoder_model, Differentiable_AriEL,\
    predefined_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                         Bidirectional, LSTM, Lambda, TimeDistributed, \
                         RepeatVector, Activation, GaussianNoise, Flatten, \
                         Reshape
                         
from keras.utils import to_categorical
import tensorflow as tf
from utils import TestActiveGaussianNoise, SelfAdjustingGaussianNoise


# TODO: move to unittest type of test


vocabSize = 3
max_senLen = 6
batchSize = 3 #4
latDim = 4
embDim = 2
                                
                                
def random_sequences_and_points(repeated=False, vocabSize=vocabSize):
    
    if not repeated:
        questions = []
        points = np.random.rand(batchSize, latDim)
        for _ in range(batchSize):
            sentence_length = max_senLen #np.random.choice(max_senLen)
            randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
            #EOS = (vocabSize+1)*np.ones(1)
            #randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, latDim)
        sentence_length = max_senLen #np.random.choice(max_senLen)
        question = np.random.choice(vocabSize, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats = [batchSize], axis=0)
        questions = np.repeat(question, repeats = [batchSize], axis=0)
        
    padded_questions = pad_sequences(questions)
    #print(questions)
    print('')
    print('padded questions')
    print(padded_questions)
    print('')
    print('points')
    print(points)
    print('')
    print('')

    return padded_questions, points


def test_DAriEL_Encoder_model():
    
    # CHECKED
    # 1. random numpy arrays pass through the encoder succesfully
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Encoding
          
          """)        

    
    questions, points = random_sequences_and_points()
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = DAriEL_Encoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim,
                                 max_senLen = max_senLen, 
                                 startId = 0)
    #print(partialModel.predict(question)[0])
    for layer in model.predict(questions):
        print(layer)
        #assert layer.shape[0] == latDim
        print('')
        #print('')

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        #print(w)
        #print('        ', g)  
        if not isinstance(g, tf.IndexedSlices):
            assert g[0] != None
    print("""
          Test fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, points)    
        
def test_DAriEL_Decoder_model():
    
    # CHECKED
    # 1. random numpy arrays pass through the encoder succesfully
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    
    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = DAriEL_Decoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim, 
                                 max_senLen = max_senLen, 
                                 startId = 0,
                                 output_type='tokens')
    
    prediction = model.predict(points)

    print(prediction)    
    
    # The batch size of predicted tokens should contain different sequences of tokens
    assert np.any(prediction[0] != prediction[1])
    

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None
    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, questions)    

   
    
    
    
        
        
def test_DAriEL_AE_dcd_model():
    
    questions, _ = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differentiable_AriEL(vocabSize = vocabSize,
                                     embDim = embDim,
                                     latDim = latDim,
                                     max_senLen = max_senLen,
                                     startId = 0,
                                     output_type = 'tokens')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions)    

    
    
    

        
        
def test_DAriEL_AE_cdc_model():
    
    _, points = random_sequences_and_points()
    
    
    print("""
          Test Auto-Encoder CDC
          
          """)        

    DAriA_cdc = Differentiable_AriEL(vocabSize = vocabSize,
                                     embDim = embDim,
                                     latDim = latDim,
                                     max_senLen = max_senLen,
                                     startId = 0,
                                     output_type = 'tokens')


    input_point = Input(shape=(latDim,), name='discrete_sequence')
    discrete_output = DAriA_cdc.decode(input_point)
    # in between some neural operations can be defined
    continuous_output = DAriA_cdc.encode(discrete_output)
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_point, outputs=continuous_output)   # + [continuous_latent_space])    
    #model.summary()
    
    for layer in model.predict(points):
        print(layer)
        print('\n')
        
    print('')
    print(points)
    print('')
    print('')
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(points, points)    




    
def test_stuff_inside_AE():
    
    
    questions, _ = random_sequences_and_points(True)
    
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    # Dense WORKS!! (it fits) but loss = 0 even for random initial weights! ERROR!!!!
    #continuous_latent_space = Dense(latDim)(continuous_latent_space)                      
    
    # GaussianNoise(stddev=.02) WORKS!! (it fits) and loss \= 0!! but stddev>=.15 
    # not always :( find reason or if it keeps happening if you restart the kernel
    #continuous_latent_space = GaussianNoise(stddev=.2)(continuous_latent_space) 
    
    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    continuous_latent_space = TestActiveGaussianNoise(stddev=.2)(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions, epochs=2)    



    



def test_noise_vs_vocabSize(vocabSize=4, std=.2):
    
    
    questions, _ = random_sequences_and_points(False, vocabSize)
    
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    #continuous_latent_space = TestActiveGaussianNoise(stddev=std)(continuous_latent_space)
    continuous_latent_space = SelfAdjustingGaussianNoise()(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        


def test_Decoder_forTooMuchNoise():
    """
    it works thanks to value clipping at the entrance of the decoder, other
    solutions might be interesting, like an interplay with layer normalization
    or allowing the latent space to be [-1,1]
    """
    
    print("""
          Test Decoding
          
          """)

    #questions, points = random_sequences_and_points()
    points = 20*np.random.rand(batchSize, latDim)
    
    
    print('points')
    print(points)
    print('')
    
    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim, 
                                 max_senLen = max_senLen, 
                                 output_type='tokens')
    
    prediction = model.predict(points)
    
    print(prediction)    
    


def test_SelfAdjustingGaussianNoise():
    
    print("""
          Test SelfAdjustingGaussianNoise
          
          """)

    ones = np.ones((1,3))
    
    inputs = Input((3,))
    type_sagn = 'scalar'  # 'tensor'  #  'ababas'   #
    output = SelfAdjustingGaussianNoise(type_sagn)(inputs)
    model = Model(inputs, output)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.summary()
    
    # it's learning to decrease its noise so it can map ones to ones
    for _ in range(10):
        model.fit(ones, ones, epochs=100, verbose=0)
        prediction = model.predict(ones)
        print(prediction)    
    

def test_DAriA_Decoder_cross_entropy():
    
    # CHECKED
    # 1. random numpy arrays pass through the encoder succesfully
    # 2. gradients /= None
    # 3. fit method works
    
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    
    
    categorical_questions = to_categorical(questions, num_classes = vocabSize)
    print('')
    #print(categorical_questions)
    print('')
    print(categorical_questions.shape)
    
    # 1. it works
    #model = vAriEL_Decoder_model(vocabSize = vocabSize, 
    #                             embDim = embDim, 
    #                             latDim = latDim, 
    #                             max_senLen = max_senLen, 
    #                             output_type='softmaxes')

    # 2. does this work?    
    DAriA_dcd = Differentiable_AriEL(vocabSize = vocabSize,
                                     embDim = embDim,
                                     latDim = latDim,
                                     max_senLen = max_senLen,
                                     startId = 0,
                                     output_type = 'softmaxes')


    input_point = Input(shape=(latDim,), name='input_point')
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(input_point)
    
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_point, outputs=discrete_output)   # + [continuous_latent_space])    
    
    #model.summary()

    
    prediction = model.predict(points)
    print('')
    print('prediction size:  ', prediction.shape)
    print('')
    print('')
    print('prediction:       ', prediction)
    print('')


    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(points, categorical_questions)    



def test_vAriEL_dcd_CCE():
    
    questions, _ = random_sequences_and_points()
    
    categorical_questions = to_categorical(questions, num_classes = vocabSize)
    print('')
    print('categorical questions')
    print('')
    print(categorical_questions)
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'softmaxes')


    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    
    continuous_latent_space = GaussianNoise('tensor')(continuous_latent_space)
    
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    model = Model(inputs=input_question, outputs=discrete_output)   # + [continuous_latent_space])    
    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
        
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        #assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(questions, categorical_questions)    







def test_vAriEL_onMNIST():
    from keras.datasets import mnist
    
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    
    print("""
          Test Auto-Encoder on MNIST
          
          """)        
    
    latDim = np.prod(input_shape)
    DAriA_cdc = Differential_AriEL(vocabSize = vocabSize,
                                   embDim = embDim,
                                   latDim = latDim,
                                   max_senLen = max_senLen,
                                   output_type = 'tokens')


    input_image = Input(shape=input_shape, name='discrete_sequence')
    input_point = Flatten()(input_image)    
    
    discrete_output = DAriA_cdc.decode(input_point)
    continuous_output = DAriA_cdc.encode(discrete_output)
    
    output_image = Reshape(input_shape)(continuous_output)
    # vocabSize + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_image, outputs=output_image)   # + [continuous_latent_space])    
    #model.summary()
    

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(x_train, x_train)    
    
    # reconstruction is perfect by construction of the representation:
    # the goal is to find a way to trim the tree


def test_DAriEL_model_from_outside():
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    
    LM = predefined_model(vocabSize, embDim)


    # it used to be vocabSize + 1 for the keras padding + 1 for EOS
    model = DAriEL_Decoder_model(vocabSize = vocabSize, 
                                 embDim = embDim, 
                                 latDim = latDim, 
                                 max_senLen = max_senLen, 
                                 startId = 0,
                                 language_model = LM,
                                 output_type='tokens')
    
    prediction = model.predict(points)

    print(prediction)
   
def test_DAriA_Decoder_wasserstein():
    """
    https://arxiv.org/pdf/1701.07875.pdf
    p4
    """
    pass





if __name__ == '__main__':
    #test_DAriEL_Decoder_model()   # works for DAriEL v2
    print('=========================================================================================')
    #test_DAriEL_Encoder_model()   # works for DAriEL v2
    print('=========================================================================================')    
    #test_DAriEL_AE_dcd_model()     # works for DAriEL v2
    print('=========================================================================================')    
    #test_DAriEL_AE_cdc_model()    # works for DAriEL v2
    print('=========================================================================================')    
    #test_stuff_inside_AE()
    print('=========================================================================================')    
    #test_Decoder_forTooMuchNoise()
    print('=========================================================================================')    
    #test_SelfAdjustingGaussianNoise()
    print('=========================================================================================')    
    #test_DAriA_Decoder_cross_entropy()   # works for DAriEL v2
    print('=========================================================================================')    
    #test_vAriEL_dcd_CCE()
    #test_new_Decoder()
    #test_vAriEL_onMNIST()
    print('=========================================================================================')    
    test_DAriEL_model_from_outside()