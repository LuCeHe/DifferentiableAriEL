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



import os
import numpy as np
from time import strftime, localtime

import keras
import keras.backend as K
from keras.legacy import interfaces
from keras.engine.base_layer import Layer
from keras.engine import InputSpec
import tensorflow as tf

class TestActiveGaussianNoise(Layer):
    @interfaces.legacy_gaussiannoise_support
    def __init__(self, stddev, **kwargs):
        super(TestActiveGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(TestActiveGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
        
    
class SelfAdjustingGaussianNoise(Layer):
    @interfaces.legacy_gaussiannoise_support
    def __init__(self, tensor_type='scalar', **kwargs):
        super(SelfAdjustingGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        
        if not tensor_type in ['scalar', 'tensor']: 
            raise ValueError("""tensor_type can be either 'scalar' or 'tensor'!""")
            
        self.tensor_type = tensor_type
        
        
        self.stddev_initializer = keras.initializers.get('ones')
        self.stddev_regularizer = keras.regularizers.get(None)
        self.stddev_constraint = keras.constraints.get(None)


        
    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.tensor_type == 'scalar':
            stddev_value = tf.Variable([1.], dtype=tf.float32)
            self.stddev = tf.ones(shape,
                                  dtype=tf.float32)
            self.stddev *= stddev_value
            self.trainable_weights = [stddev_value]
            
        else:
            self.stddev = self.add_weight(shape=shape,
                                         initializer=self.stddev_initializer,
                                         regularizer=self.stddev_regularizer,
                                         constraint=self.stddev_constraint,
                                         name='gamma',
                                         )
        super(SelfAdjustingGaussianNoise, self).build(input_shape)

    def call(self, inputs, training=None):
        def noised():
            return inputs + self.stddev*K.random_normal(shape=K.shape(inputs),
                                                        mean=0.,
                                                        stddev=1.)
        return K.in_train_phase(noised, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SelfAdjustingGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



def make_directories(time_string = None):
    
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
    plt.title('softmax evolution during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')
        
    
    
    
    
def checkDuringTraining(generator_class, indices_sentences, encoder_model, decoder_model, batchSize, latDim):
    
    
    print('')
    print('original sentences')
    print('')

    sentences = generator_class.indicesToSentences(indices_sentences)

    print(sentences)
    
    print('')
    print('reconstructed sentences')
    print('')

    point = encoder_model.predict(indices_sentences)
    indices_reconstructed, _ = decoder_model.predict(point)    

    sentences_reconstructed = generator_class.indicesToSentences(indices_reconstructed)
    
    print(sentences_reconstructed)

    print('')
    print('generated sentences')
    print('')

    noise = np.random.rand(batchSize, latDim)
    indicess, softmaxes = decoder_model.predict(noise)
    print(indicess)
    print(softmaxes)
    print('\n\n\n')
    print('HERE!!')
    sentences_generated = generator_class.indicesToSentences(indicess)

    print(sentences_generated)
    print('')
    print(softmaxes[0][0])
    print('')
    
    return softmaxes

    