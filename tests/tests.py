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

#from DifferentiableAriEL.nnets.AriEL import DAriEL_Encoder_model, DAriEL_Decoder_model, Differentiable_AriEL, \
#    predefined_model, DAriEL_Encoder_Layer, DAriEL_Decoder_Layer

# TODO: move to unittest type of test

vocab_size = 3
maxlen = 6
batch_size = 3  # 4
lat_dim = 4
emb_dim = 2

                                
def biasedSequences(batch_size=3):
    
    options = [[0, 1, 0, 0],
               [0, 2, 0, 0],
               [0, 1, 1, 0],
               [0, 1, 2, 0],
               [0, 1, 3, 0],
               [0, 2, 3, 0],
               [0, 1, 2, 3],
               [0, 2, 3, 3]]
    
    options = np.array(options)

    frequencies = [1, 1, 6, 6, 11, 6, 1, 1]
    probs = [f / sum(frequencies) for f in frequencies]
    
    biased_data_indices = np.random.choice(len(options), batch_size, probs).tolist()
    biased_sequences = options[biased_data_indices]

    return biased_sequences
    
                                    
def random_sequences_and_points(batch_size=3, lat_dim=4, maxlen=6, repeated=False, vocab_size=vocab_size):
    
    if not repeated:
        questions = []
        points = np.random.rand(batch_size, lat_dim)
        for _ in range(batch_size):
            sentence_length = maxlen  # np.random.choice(maxlen)
            randomQ = np.random.choice(vocab_size, sentence_length)  # + 1
            # EOS = (vocab_size+1)*np.ones(1)
            # randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, lat_dim)
        sentence_length = maxlen  # np.random.choice(maxlen)
        question = np.random.choice(vocab_size, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats=[batch_size], axis=0)
        questions = np.repeat(question, repeats=[batch_size], axis=0)
        
    padded_questions = pad_sequences(questions)

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
    
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = DAriEL_Encoder_model(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 startId=0)
    # print(partialModel.predict(question)[0])
    for layer in model.predict(questions):
        print(layer)
        # assert layer.shape[0] == lat_dim
        print('')
        # print('')

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        # print(w)
        # print('        ', g)  
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
    
    # it used to be vocab_size + 1 for the keras padding + 1 for EOS
    model = DAriEL_Decoder_model(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 startId=0,
                                 output_type='tokens')
    
    prediction = model.predict(points)

    print(prediction)    
    
    # The batch size of predicted tokens should contain different sequences of tokens
    assert np.any(prediction[0] != prediction[1])

    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        # assert g[0] != None
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

    DAriA_dcd = Differentiable_AriEL(vocab_size=vocab_size,
                                     emb_dim=emb_dim,
                                     lat_dim=lat_dim,
                                     maxlen=maxlen,
                                     startId=0,
                                     output_type='tokens')

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)  # + [continuous_latent_space])    
    model.summary()
    
    predictions = model.predict(questions)
    from prettytable import PrettyTable
    t = PrettyTable(['Question', 'Reconstruction'])
    for a in zip(questions, predictions):
        t.add_row([*a])
    
    print(t)
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        # assert g[0] != None

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

    DAriA_cdc = Differentiable_AriEL(vocab_size=vocab_size,
                                     emb_dim=emb_dim,
                                     lat_dim=lat_dim,
                                     maxlen=maxlen,
                                     startId=0,
                                     output_type='tokens')

    input_point = Input(shape=(lat_dim,), name='discrete_sequence')
    discrete_output = DAriA_cdc.decode(input_point)
    # in between some neural operations can be defined
    continuous_output = DAriA_cdc.encode(discrete_output)
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_point, outputs=continuous_output)  # + [continuous_latent_space])    
    # model.summary()
    
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
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)
        # assert g[0] != None

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

    DAriA_dcd = Differential_AriEL(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   lat_dim=lat_dim,
                                   maxlen=maxlen,
                                   output_type='tokens')

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    # Dense WORKS!! (it fits) but loss = 0 even for random initial weights! ERROR!!!!
    # continuous_latent_space = Dense(lat_dim)(continuous_latent_space)
    
    # GaussianNoise(stddev=.02) WORKS!! (it fits) and loss \= 0!! but stddev>=.15 
    # not always :( find reason or if it keeps happening if you restart the kernel
    # continuous_latent_space = GaussianNoise(stddev=.2)(continuous_latent_space) 
    
    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    continuous_latent_space = TestActiveGaussianNoise(stddev=.2)(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)  # + [continuous_latent_space])    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        # assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(questions, questions, epochs=2)    


def test_noise_vs_vocab_size(vocab_size=4, std=.2):
    
    questions, _ = random_sequences_and_points(False, vocab_size)
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   lat_dim=lat_dim,
                                   maxlen=maxlen,
                                   output_type='tokens')

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)

    # testActiveGaussianNoise(stddev=.02) WORKS!! but not enough at test time :()
    # continuous_latent_space = TestActiveGaussianNoise(stddev=std)(continuous_latent_space)
    continuous_latent_space = SelfAdjustingGaussianNoise()(continuous_latent_space)

    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_question, outputs=discrete_output)  # + [continuous_latent_space])    
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

    # questions, points = random_sequences_and_points()
    points = 20 * np.random.rand(batch_size, lat_dim)
    
    print('points')
    print(points)
    print('')
    
    # it used to be vocab_size + 1 for the keras padding + 1 for EOS
    model = vAriEL_Decoder_model(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 output_type='tokens')
    
    prediction = model.predict(points)
    
    print(prediction)    


def test_SelfAdjustingGaussianNoise():
    
    print("""
          Test SelfAdjustingGaussianNoise
          
          """)

    ones = np.ones((1, 3))
    
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
    
    categorical_questions = to_categorical(questions, num_classes=vocab_size)
    print('')
    # print(categorical_questions)
    print('')
    print(categorical_questions.shape)
    
    # 1. it works
    # model = vAriEL_Decoder_model(vocab_size = vocab_size,
    #                             emb_dim = emb_dim,
    #                             lat_dim = lat_dim,
    #                             maxlen = maxlen,
    #                             output_type='softmaxes')

    # 2. does this work?    
    DAriA_dcd = Differentiable_AriEL(vocab_size=vocab_size,
                                     emb_dim=emb_dim,
                                     lat_dim=lat_dim,
                                     maxlen=maxlen,
                                     startId=0,
                                     output_type='softmaxes')

    input_point = Input(shape=(lat_dim,), name='input_point')
    # in between some neural operations can be defined
    discrete_output = DAriA_dcd.decode(input_point)
    
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_point, outputs=discrete_output)  # + [continuous_latent_space])    
    
    # model.summary()
    
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
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        # assert g[0] != None

    print("""
          Test Fit
          
          """)
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(points, categorical_questions)    


def test_vAriEL_dcd_CCE():
    
    questions, _ = random_sequences_and_points()
    
    categorical_questions = to_categorical(questions, num_classes=vocab_size)
    print('')
    print('categorical questions')
    print('')
    print(categorical_questions)
    
    print("""
          Test Auto-Encoder DCD
          
          """)        

    DAriA_dcd = Differential_AriEL(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   lat_dim=lat_dim,
                                   maxlen=maxlen,
                                   output_type='softmaxes')

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA_dcd.encode(input_question)
    
    continuous_latent_space = GaussianNoise('tensor')(continuous_latent_space)
    
    discrete_output = DAriA_dcd.decode(continuous_latent_space)
    model = Model(inputs=input_question, outputs=discrete_output)  # + [continuous_latent_space])    
    
    model.summary()
    
    for layer in model.predict(questions):
        print(layer)
        print('\n')
    
    print("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        # assert g[0] != None

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
    
    lat_dim = np.prod(input_shape)
    DAriA_cdc = Differential_AriEL(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   lat_dim=lat_dim,
                                   maxlen=maxlen,
                                   output_type='tokens')

    input_image = Input(shape=input_shape, name='discrete_sequence')
    input_point = Flatten()(input_image)    
    
    discrete_output = DAriA_cdc.decode(input_point)
    continuous_output = DAriA_cdc.encode(discrete_output)
    
    output_image = Reshape(input_shape)(continuous_output)
    # vocab_size + 1 for the keras padding + 1 for EOS
    model = Model(inputs=input_image, outputs=output_image)  # + [continuous_latent_space])    
    # model.summary()

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
    
    LM = predefined_model(vocab_size, emb_dim)

    # it used to be vocab_size + 1 for the keras padding + 1 for EOS
    model = DAriEL_Decoder_model(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 startId=0,
                                 language_model=LM,
                                 output_type='both')
    
    prediction = model.predict(points)
    
    print(prediction)


def test_DAriEL_model_from_outside_v2():
    
    print("""
          Test Decoding
          
          """)

    questions, points = random_sequences_and_points()
    answers = to_categorical(questions[:, 1], vocab_size)
    
    LM = predefined_model(vocab_size, emb_dim)
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
    
    LM.fit(questions, answers, epochs=100)    

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

    print(indicess)
    

def mini_test():
    
    batch_size = 2
    lat_dim, curDim, vocab_size = 4, 2, 7
    print('Parameters:\nlat_dim = {}\ncurDim = {}\nvocab_size = {}\nbatch_size = {}\n\n'.format(lat_dim, curDim, vocab_size, batch_size))
    
    unfolding_point = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, lat_dim))
    softmax_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, vocab_size))
    t_lat_dim = tf.shape(unfolding_point)[-1]
    
    expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
    t_curDim = tf.constant(curDim) + 1

    x = expanded_unfolding_point[:, :, curDim]
    t_x = expanded_unfolding_point[:, :, t_curDim]  # works!

    one_hots = dynamic_one_hot(softmax_placeholder, lat_dim, curDim)
    t_one_hots = dynamic_one_hot(softmax_placeholder, t_lat_dim, t_curDim)

    # run
    sess = tf.compat.v1.Session()
    
    random_4softmax = np.random.rand(batch_size, vocab_size)
    
    lines = []
    for line in random_4softmax:
        index = np.random.choice(vocab_size)
        line[index] = 100
        lines.append(line)
    
    random_4softmax = np.array(lines)
    sum_r = random_4softmax.sum(axis=1, keepdims=True)
    initial_softmax = random_4softmax / sum_r
    initial_point = np.random.rand(batch_size, lat_dim)
    feed_data = {softmax_placeholder: initial_softmax, unfolding_point: initial_point}
    results = sess.run([expanded_unfolding_point, x, t_x], feed_data)
    
    for r in results:
        print(r)
        print('\n\n')


def test_DAriA_Decoder_wasserstein():
    """
    https://arxiv.org/pdf/1701.07875.pdf
    p4
    """
    pass


def checkSpaceCoverageDecoder(decoder_model, lat_dim, maxlen):
    _, points = random_sequences_and_points(batch_size=10000, lat_dim=lat_dim, maxlen=maxlen)
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


def checkSpaceCoverageEncoder(encoder_model, lat_dim, maxlen):
    
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

    
def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def test_2d_visualization_trainInside():
    
    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough
    
    maxlen = 4
    vocab_size = 4
    emb_dim = int(np.sqrt(vocab_size) + 1)
    lat_dim = 2
    epochs = 100
    
    DAriA = Differentiable_AriEL(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 output_type='both',
                                 startId=0)
    
    input_questions = Input(shape=(lat_dim,), name='question')
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point[0])

    checkSpaceCoverageDecoder(decoder_model, lat_dim, maxlen)
    
    # bias the representation
    
    bs = biasedSequences(batch_size=1000)
    categorical_bs = to_categorical(bs, num_classes=vocab_size)

    bs_val = biasedSequences(batch_size=100)
    categorical_bs_val = to_categorical(bs_val, num_classes=vocab_size)
    
    print("""
          Test Auto-Encoder DCD
          
          """)

    input_question = Input(shape=(None,), name='discrete_sequence')
    continuous_latent_space = DAriA.encode(input_question)    
    discrete_output = DAriA.decode(continuous_latent_space)
    
    ae_model = Model(inputs=input_question, outputs=discrete_output[1])  # + [continuous_latent_space])     
    ae_model.summary()    

    print("""
          Test Fit
          
          """)
    
    time_string = timeStructured()
    tensorboard = TensorBoard(log_dir="logs/{}_test_2d_visualization".format(time_string), histogram_freq=int(epochs / 10), write_grads=True)
    callbacks = []  # [tensorboard]
    ae_model.compile(loss='mse', optimizer='sgd', run_eagerly=False)
    ae_model.fit(bs, categorical_bs, epochs=epochs, callbacks=callbacks, validation_data=[bs_val, categorical_bs_val], validation_freq=1)    

    checkSpaceCoverageDecoder(decoder_model, lat_dim, maxlen)
    
    predictions = ae_model.predict(bs)
    pred = np.argmax(predictions, axis=1)
        
    t = PrettyTable(['bs', 'pred'])
    for a in zip(bs, pred):
        t.add_row([*a])
    
    print(t)



def test():
    
    maxlen = 10  # 20 #
    vocab_size = 100  # 1500 #
    emb_dim = int(np.sqrt(vocab_size) + 1)
    lat_dim = 5
    epochs = 10
    
    questions, _ = random_sequences_and_points(batch_size=10, lat_dim=lat_dim, maxlen=maxlen, vocab_size=vocab_size)
    answers = to_categorical(questions[:, 1], vocab_size)
    logger.info(answers)
    
    LM = predefined_model(vocab_size, emb_dim)
    LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
    LM.fit(questions, answers, epochs=epochs)    
    
    DAriA = Differentiable_AriEL(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=maxlen,
                                 output_type='both',
                                 language_model=LM,
                                 PAD=0)
    
    input_questions = Input(shape=(lat_dim,), name='question')
    dense = Dense(4)(input_questions)
    point = DAriA.decode(dense)
    decoder_model = Model(inputs=input_questions, outputs=point[1])

    _, points = random_sequences_and_points(batch_size=100, lat_dim=lat_dim, maxlen=maxlen)
    pred = decoder_model.predict(points, verbose=1)
    
    logger.info(pred)
    
    showGradientsAndTrainableParams(decoder_model)


def test_2_old():
    vocab_size = 5  # 1500 #
    emb_dim = 2
    lat_dim = 4
    max_length = 10
    epochs = 10
    batch_size = 3
    
    DAriA = Differentiable_AriEL(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=max_length,
                                 output_type='both',
                                 language_model=None,
                                 PAD=0)
    
    input_questions = Input(shape=(lat_dim,), name='question')
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point[0])
    
    _, points = random_sequences_and_points(batch_size=batch_size, lat_dim=lat_dim, maxlen=max_length)
    pred_output = decoder_model.predict(points, verbose=1)
    
    logger.info(pred_output)

    
def test_2_tf():
    
    vocab_size = 6  # 1500 #
    emb_dim = 4
    lat_dim = 2
    max_length = 7
    batch_size = 3
    PAD = 0
    
    DAriA = Differentiable_AriEL(vocab_size=vocab_size,
                                 emb_dim=emb_dim,
                                 lat_dim=lat_dim,
                                 maxlen=max_length,
                                 output_type='both',
                                 language_model=None,
                                 tf_RNN=True,
                                 PAD=PAD)
    
    input_questions = Input(shape=(lat_dim,), name='question')
    point = DAriA.decode(input_questions)
    decoder_model = Model(inputs=input_questions, outputs=point)
    
    _, points = random_sequences_and_points(batch_size=batch_size, lat_dim=lat_dim, maxlen=max_length)
    pred_output = decoder_model.predict(points, batch_size=batch_size, verbose=1)
    
    logger.warn(np.argmax(pred_output, axis=2))
    
    for p in pred_output:
        print(p.shape)



def finetuning():

    vocab_size = 6  # 1500 #
    emb_dim = 5
    lat_dim = 2
    max_length = 4
    batch_size = 3
    PAD = 0

    language_model = predefined_model(vocab_size, emb_dim)

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    _, inputs = random_sequences_and_points(batch_size=batch_size, lat_dim=lat_dim, maxlen=max_length)
    
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None, lat_dim))

    curDim = tf.zeros(1)
    timeStep = tf.zeros(1)
    b = tf.shape(inputs_placeholder)[0]
    one_softmax, unfolding_point = tf.zeros([b, vocab_size]), tf.zeros([b, lat_dim])
    tokens = tf.zeros([b, max_length])

    b = tf.shape(one_softmax)[0]
    curDimVector = tf.tile(curDim[tf.newaxis,:], [b, 1])
    timeStepVector = tf.tile(timeStep[tf.newaxis,:], [b, 1])

    state = one_softmax, tokens, unfolding_point, curDimVector, timeStepVector
    
    input_point = inputs_placeholder
    
    all_results = []
    for _ in tqdm(range(max_length)):
        
        input_point = input_point
        one_softmax, tokens, unfolding_point, curDimVector, timeStepVector = state
        curDim = curDimVector[0]
        timeStep = timeStepVector[0]
        
        # initialization        
        PAD_layer = Input(tensor=PAD*tf.ones_like(input_point[:,0, tf.newaxis]))
        initial_softmax = language_model(PAD_layer)
        
        # FIXME: it would be interesting to consider what would happen if we feed different
        # points within a batch
        pred_t = tf.reduce_mean(timeStep) > 0  # tf.math.greater_equal(zero, timeStep)
        
        unfolding_point = tf.cond(pred_t, lambda: input_point, lambda: unfolding_point)
        one_softmax = tf.cond(pred_t, lambda: initial_softmax, lambda: one_softmax)
        #tokens = tf.cond(pred_t, lambda: start_layer, lambda: tokens, name='tokens')
        
        token, unfolding_point = pzToSymbolAndZ([one_softmax, unfolding_point, curDim])
        token.set_shape((None,1))
        token = tf.squeeze(token, axis=1)    
        tokens = replace_column(tokens, token, timeStep)
        
        # get the softmax for the next iteration
        # make sure you feed only up to the tokens that have been produced now ([:timeStep]
        # otherwise you are feeding a sentence with tons of zeros at the end. 
        tokens_in = Input(tensor=tokens[:, :tf.cast(tf.squeeze(timeStep), dtype=tf.int64)+1])
        one_softmax = language_model(tokens_in)        
        
        # NOTE: at each iteration, change the dimension, and add a timestep
        lat_dim = tf.cast(tf.shape(unfolding_point)[-1], dtype=tf.float32)
        pred_l = tf.reduce_mean(curDim) + 1 >= tf.reduce_mean(lat_dim)  # tf.math.greater_equal(curDim, lat_dim)
        curDim = tf.cond(pred_l, lambda: tf.zeros_like(curDim), lambda: tf.add(curDim, 1), name='curDim')
        timeStep = tf.add(timeStep, 1)
        
        b = tf.shape(one_softmax)[0]
        curDimVector = tf.tile(curDim[tf.newaxis,:], [b, 1])
        timeStepVector = tf.tile(timeStep[tf.newaxis,:], [b, 1])
        
        output = [one_softmax, curDim, timeStep, curDimVector, timeStepVector]
        state = [one_softmax, tokens, unfolding_point, curDimVector, timeStepVector]

        feed_data = {inputs_placeholder: inputs}
        results = sess.run(output, feed_data)  # ([output, state], feed_data)
        results = results #+ [results[-1].shape]
        all_results.append([result.shape for result in results])
        
    t = PrettyTable()
    for a in all_results:
        t.add_row([*a])
    
    print(t)
