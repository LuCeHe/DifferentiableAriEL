import logging
import os
import sys

import numpy
from prettytable import PrettyTable

from GenericTools.LeanguageTreatmentTools.sentence_generators import GzipToNextStepGenerator, GzipToIndicesGenerator

numpy.set_printoptions(threshold=sys.maxsize, suppress=True, precision=3)

import numpy as np

import matplotlib

matplotlib.use('TkAgg')

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from keras.backend.tensorflow_backend import set_session

from DifferentiableAriEL.nnets.AriEL import AriEL

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
# from GenericTools.SacredTools.VeryCustomSacred import CustomFileStorageObserver

from DifferentiableAriEL.convenience_tools.utils import train_language_model
from GenericTools.LeanguageTreatmentTools.nlp import Vocabulary
from GenericTools.StayOrganizedTools.utils import timeStructured

ex = Experiment('LSNN_AE')
ex.observers.append(FileStorageObserver.create("experiments"))
# ex.observers.append(CustomFileStorageObserver.create("../data/experiments"))

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

CDIR = os.path.dirname(os.path.realpath(__file__))

CDIR_, _ = os.path.split(CDIR)

@ex.config
def cfg():
    time_string = timeStructured()

    # GPU setting

    GPU = 1
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
    log_path = temp_folder + 'logs/' + time_string + '_'

    data_dir = os.path.join(CDIR_, 'data')
    grammar_filepath = os.path.join(data_dir,'simplerREBER_grammar.cfg')
    train_gzip = os.path.join(data_dir, 'REBER_biased_train.gz')
    val_gzip = os.path.join(data_dir, 'REBER_biased_val.gz')

    # params

    maxlen = 200
    encoder_type = 0
    decoder_type = 0
    size_lat_dim = 1e6

    lat_dim = 16  # 2

    # training params

    is_laptop = True
    if is_laptop:
        batch_size = 3  # 256
        nb_lines = 5
        epochs = 10
    else:
        batch_size = 256
        nb_lines = 1e6
        epochs = 100

    do_train = True

    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    vocab_size = vocabulary.getMaxVocabularySize()
    emb_dim = int(np.sqrt(vocab_size) + 1)
    units = 256
    del vocabulary

    training_params = {}
    training_params.update(
        train_gzip=train_gzip,
        val_gzip=val_gzip,
        grammar_filepath=grammar_filepath,
        batch_size=batch_size,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        units=units,
        epochs=epochs,
        nb_lines=nb_lines,
        LM_path=LM_path,
        log_path=log_path)


@ex.capture
def checkGeneration(
        LM,
        lat_dim,
        vocab_size,
        maxlen,
        encoder_type,
        decoder_type,
        size_lat_dim,
        grammar_filepath):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    PAD = vocabulary.padIndex

    points = np.random.rand(100, lat_dim)

    ariel = AriEL(
        vocab_size=vocab_size,
        lat_dim=lat_dim,
        maxlen=maxlen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_lat_dim=size_lat_dim,
        PAD=PAD
    )

    input_point = Input(shape=(lat_dim,), name='question')
    point = ariel.decode(input_point)
    decoder_model = Model(inputs=input_point, outputs=point)

    prediction = decoder_model.predict(points).astype(int)

    print(prediction)
    sentences = vocabulary.indicesToSentences(prediction)

    print(sentences)


@ex.capture
def checkRandomReconstruction(
        grammar_filepath,
        LM,
        lat_dim,
        vocab_size,
        emb_dim,
        maxlen,
        encoder_type,
        decoder_type,
        size_lat_dim,
        batch_size):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    PAD = vocabulary.padIndex

    ariel = AriEL(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        lat_dim=lat_dim,
        maxlen=maxlen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_lat_dim=size_lat_dim,
        PAD=PAD
    )

    input_questions = Input(shape=(None,), name='question')
    continuous_output = ariel.encode(input_questions)
    discrete_output = ariel.decode(continuous_output)
    reconstruction_model = Model(inputs=input_questions, outputs=discrete_output)

    sentences = np.random.randint(vocab_size, size=(batch_size, maxlen))
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
        grammar_filepath,
        sentences,
        lat_dim,
        vocab_size,
        emb_dim,
        maxlen,
        encoder_type,
        decoder_type,
        size_lat_dim):
    vocabulary = Vocabulary.fromGrammarFile(grammar_filepath)
    PAD = vocabulary.padIndex

    sentences = sentences[:, :maxlen]
    maxlen = sentences.shape[1]
    ariel = AriEL(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        lat_dim=lat_dim,
        maxlen=maxlen,
        output_type='tokens',
        language_model=LM,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        size_lat_dim=size_lat_dim,
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
def test_on_reber(
        train_gzip,
        val_gzip,
        grammar_filepath,
        batch_size,
        LM_path,
        epochs,
        training_params,
        do_train,
        _log):

    # FIXME: it's cool that it is learning but it doesn't
    # seem to be learning enough
    if not os.path.isfile(LM_path) or do_train:
        LM = train_language_model(train_method='transformer', **training_params)
    else:
        LM = load_model(LM_path)

    """
    print('\n   Check LM   \n')

    generator = GzipToNextStepGenerator(val_gzip, grammar_filepath, batch_size)

    batch = next(generator)
    output = LM.predict(batch[0])

    t = PrettyTable(['batch', 'pred'])
    for a in zip(batch[0], output):
        t.add_row([*a])

    print(t)

    print('\n   Check AriEL Decoder   \n')

    checkGeneration(LM)

    print('\n   Check AriEL Random Data Reconstruction   \n')

    for LanMod in [None, LM]:
        checkRandomReconstruction(LM=LanMod)

    print('\n   Check AriEL Training Data Reconstruction   \n')

    generator = GzipToIndicesGenerator(val_gzip, grammar_filepath, batch_size)
    sentences = next(generator)
    for LanMod in [None, LM]:
        checkTrainingReconstruction(LM=LanMod, sentences=sentences)
        
    """
