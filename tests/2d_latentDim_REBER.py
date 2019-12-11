
import os, sys, copy, logging, pathlib, gzip, time
from time import strftime, localtime
from prettytable import PrettyTable

import numpy
numpy.set_printoptions(threshold=sys.maxsize, suppress=True, precision=3)

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import load_model                        
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from keras.backend.tensorflow_backend import set_session

from DifferentiableAriEL.language.nlp import Vocabulary
from DifferentiableAriEL.language.sentenceGenerators import generateFromGzip, \
    SentenceToIndicesGenerator, IndicesToNextStepGenerator, GzipToNextStepGenerator, \
    GzipToIndicesGenerator
from DifferentiableAriEL.language.utils import timeStructured
from DifferentiableAriEL.nnets.AriEL import AriEL
from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model

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
ex.observers.append(CustomFileStorageObserver.create("../data/experiments"))

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

    time_string = timeStructured()    
    
    # GPU setting
    
    GPU = 0
    GPU_fraction = .80
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction
    set_session(tf.Session(config=config))

    # paths

    temp_folder = '../data/tmp/'
    if not os.path.isdir(temp_folder): os.mkdir(temp_folder)

    LM_path = temp_folder + 'LM_model_REBER.h5'
    logPath = temp_folder + 'logs/' + time_string + '_'

    grammar_filepath = '../data/simplerREBER_grammar.cfg'
    gzip_filepath = '../data/REBER_biased_train.gz'
    
    # params
    
    max_senLen = 200
    encoder_type = 0
    decoder_type = 0
    size_latDim = 1e6

    latDim = 16  # 2
        
    epochs = 10
    steps_per_epoch = 1e3
    batch_size = 256
        
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    vocabSize = vocabulary.getMaxVocabularySize()
    
    embDim = int(np.sqrt(vocabSize) + 1)
    
    PAD = vocabulary.padIndex
    START = vocabulary.startIndex
    END = vocabulary.endIndex
    
    choices = [[START, END],
               [START, END],
               [START, END],
               [START, END]
               ]
    
    del vocabulary


@ex.capture
def checkGeneration(
        LM,
        latDim,
        vocabSize,
        PAD,
        embDim,
        max_senLen,
        encoder_type,
        decoder_type,
        size_latDim,
        grammar_filepath
        ):  

    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    
    points = np.random.rand(100, latDim)
    
    ariel = AriEL(
        vocabSize=vocabSize,
        embDim=embDim,
        latDim=latDim,
        max_senLen=max_senLen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_latDim=size_latDim,
        PAD=PAD
        )
    
    input_point = Input(shape=(latDim,), name='question')
    point = ariel.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=point)
    
    prediction = decoder_model.predict(points).astype(int)
    
    print(prediction)
    sentences = vocabulary.indicesToSentences(prediction)
    
    print(sentences)


@ex.capture
def checkRandomReconstruction(
        LM,
        latDim,
        vocabSize,
        PAD,
        embDim,
        choices,
        max_senLen,
        encoder_type,
        decoder_type,
        size_latDim,
        batch_size
        ):

    ariel = AriEL(
        vocabSize=vocabSize,
        embDim=embDim,
        latDim=latDim,
        max_senLen=max_senLen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_latDim=size_latDim,
        PAD=PAD
        )
    
    input_questions = Input(shape=(None,), name='question')    
    continuous_output = ariel.encode(input_questions)
    discrete_output = ariel.decode(continuous_output)
    reconstruction_model = Model(inputs=input_questions, outputs=discrete_output)
    
    sentences = np.random.randint(vocabSize, size=(batch_size, max_senLen))
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
def checkTrainingReconstruction(
        LM,
        sentences,
        latDim,
        vocabSize,
        PAD,
        embDim,
        choices,
        max_senLen,
        encoder_type,
        decoder_type,
        size_latDim
        ):
    
    sentences = sentences[:, :max_senLen]
    max_senLen = sentences.shape[1]
    ariel = AriEL(
        vocabSize=vocabSize,
        embDim=embDim,
        latDim=latDim,
        max_senLen=max_senLen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_latDim=size_latDim,
        PAD=PAD
        )
    
    input_questions = Input(shape=(None,), name='question')    
    continuous_output = ariel.encode(input_questions)
    discrete_output = ariel.decode(continuous_output)
    reconstruction_model = Model(inputs=input_questions, outputs=discrete_output)
    
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
    
    
@ex.automain    
def test_2d_visualization_trainOutside(
        gzip_filepath,
        grammar_filepath,
        batch_size,
        vocabSize,
        embDim,
        LM_path, epochs, steps_per_epoch,
        logPath,
        _log
        ):
    
    generator = GzipToNextStepGenerator(gzip_filepath, grammar_filepath, batch_size)

    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough
    if not os.path.isfile(LM_path):
        LM = predefined_model(vocabSize, embDim)
        LM.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['categorical_accuracy'])
        callbacks = []
        callbacks.append(TensorBoard(
            logPath,
            histogram_freq=int(epochs / 20) + 1,
            write_graph=False,
            write_grads=True,
            write_images=False,
            batch_size=10
            ))

        LM.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
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

    checkGeneration(LM)

    print('\n   Check AriEL Random Data Reconstruction   \n')
    
    # for LanMod in [None, LM]:
    #    checkRandomReconstruction(LM=LanMod)

    print('\n   Check AriEL Training Data Reconstruction   \n')
        
    # generator = GzipToIndicesGenerator(gzip_filepath, grammar_filepath, batch_size)
    # sentences = next(generator)
    # for LanMod in [None, LM]:
    #    checkTrainingReconstruction(LM=LanMod, sentences=sentences)
