import tensorflow as tf
from GenericTools.SacredTools.CustomSacred import CustomExperiment
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.client import timeline

from DifferentiableAriEL.nnets.AriEL import AriEL
from GenericTools.LeanguageTreatmentTools.random import random_sequences_and_points

ex = CustomExperiment('2d')


@ex.config
def cfg():
    batchSize = 3,
    latDim = 3,
    max_senLen = 6,
    vocabSize = 2


@ex.automain
def main_test(
        vocabSize,
        max_senLen,
        embDim,
        latDim,
        batchSize):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    sentences, points = random_sequences_and_points(
        batchSize=batchSize,
        latDim=latDim,
        max_senLen=max_senLen,
        vocabSize=vocabSize)

    tf_sentences = tf.convert_to_tensor(sentences)
    tf_points = tf.convert_to_tensor(points)
    for model_type in range(4):
        DAriA = AriEL(
            vocabSize=vocabSize,
            embDim=embDim,
            latDim=latDim,
            max_senLen=max_senLen,
            output_type='both',
            language_model=None,
            encoder_type=model_type,
            decoder_type=model_type,
            PAD=0
        )

        input_questions = Input(tensor=tf_sentences, name='question')
        continuous_output = DAriA.encode(input_questions)
        encoder_model = Model(inputs=input_questions, outputs=continuous_output)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(3):
                sess.run(encoder_model,
                         options=options,
                         run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_Etype_{}_step_{}.json'.format(model_type, i), 'w') as f:
                    f.write(chrome_trace)

        input_point = Input(tensor=tf_points, name='question')
        point = DAriA.decode(input_point)
        decoder_model = Model(inputs=input_point, outputs=point)
