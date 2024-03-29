import logging

import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter
from sacred.utils import apply_backspaces_and_linefeeds
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.client import timeline

from DifferentiableAriEL.nnets.AriEL import AriEL
from GenericTools.LeanguageTreatmentTools.random import random_sequences_and_points

ex = Experiment('ED_versions')
ex.observers.append(FileStorageObserver.create("experiments"))
# ex.observers.append(CustomFileStorageObserver.create("experiments"))

# ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds

# set up a custom logger
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.setLevel('INFO')

# attach it to the experiment
ex.logger = logger


# ex = CustomExperiment('ED_versions')


@ex.config
def cfg():
    batch_size = 4
    lat_dim = 3
    maxlen = 6
    vocab_size = 2
    emb_dim = 1
    n_profiles = 3


@ex.automain
@LogFileWriter(ex)
def main_test(
        vocab_size,
        maxlen,
        emb_dim,
        lat_dim,
        batch_size,
        n_profiles,
        _log):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    sentences, points = random_sequences_and_points(
        batch_size=batch_size,
        lat_dim=lat_dim,
        maxlen=maxlen,
        vocab_size=vocab_size)

    tf_sentences = tf.convert_to_tensor(sentences, dtype=tf.float32)
    tf_points = tf.convert_to_tensor(points, dtype=tf.float32)
    for model_type in range(0, 3):
        DAriA = AriEL(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            lat_dim=lat_dim,
            maxlen=maxlen,
            output_type='both',
            language_model=None,
            encoder_type=model_type,
            decoder_type=model_type,
            PAD=0
        )

        _log.info('\n##########################################')
        _log.info('           AriEL Encoder {}'.format(model_type))
        _log.info('##########################################\n')
        input_questions = Input(tensor=tf_sentences, name='question')
        continuous_output = DAriA.encode(input_questions)
        encoder_model = Model(inputs=input_questions, outputs=continuous_output)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(n_profiles):
                sess.run(continuous_output,
                         options=options,
                         run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                destination_json = 'experiments/timeline_Etype_{}_step_{}.json'.format(model_type, i)
                with open(destination_json, 'w') as f:
                    f.write(chrome_trace)
                ex.add_artifact(destination_json)

        _log.info('\n##########################################')
        _log.info('           AriEL Decoder {}'.format(model_type))
        _log.info('##########################################\n')

        input_point = Input(tensor=tf_points, name='question')
        point = DAriA.decode(input_point)
        decoder_model = Model(inputs=input_point, outputs=point)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(n_profiles):
                sess.run(point,
                         options=options,
                         run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                destination_json = 'experiments/timeline_Dtype_{}_step_{}.json'.format(model_type, i)
                with open(destination_json, 'w') as f:
                    f.write(chrome_trace)
                ex.add_artifact(destination_json)
