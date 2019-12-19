import logging

import nltk
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

import grammar_on_transformer.dataloader as dd
from GenericTools.KerasTools.convenience_layers import predefined_model
from GenericTools.LeanguageTreatmentTools.sentence_generators import GzipToNextToken_KerasGenerator, \
    generateFromGzip
from grammar_on_transformer.layers.transformer import Transformer

logger = logging.getLogger(__name__)


def train_language_model(train_method='transformer', **training_params):
    if train_method == 'LSTM':
        LM = with_lstm(**training_params)
    elif train_method == 'LSTM with curriculum learning':
        LM = with_lstm_curriculum_learning(**training_params)
    elif train_method == 'transformer':
        LM = with_transformer(**training_params)
    else:
        raise NotImplementedError

    return LM


def with_lstm(
        generator,
        vocab_size,
        emb_dim,
        epochs,
        steps_per_epoch,
        LM_path,
        log_path):
    LM = predefined_model(vocab_size, emb_dim)
    LM.compile(
        loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=['categorical_accuracy'])

    callbacks = []

    callbacks.append(TensorBoard(
        log_path,
        histogram_freq=int(epochs / 20) + 1,
        write_graph=False,
        write_grads=True,
        write_images=False,
        batch_size=10
    ))

    LM.fit_generator(
        generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks)
    LM.save(LM_path)
    return LM


def with_lstm_curriculum_learning(
        train_gzip,
        val_gzip,
        grammar_filepath,
        batch_size,
        vocab_size,
        emb_dim,
        units,
        epochs,
        nb_lines,
        LM_path,
        log_path):
    LM = predefined_model(vocab_size, emb_dim, units)
    LM.summary()
    LM.compile(
        loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=['categorical_accuracy'])

    callbacks = []
    """
    tb = TensorBoard(
        log_path,
        histogram_freq=int(epochs / 20) + 1,
        write_graph=False,
        write_grads=True,
        write_images=False,
        batch_size=10
    )
    """
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(1 * epochs / 4))
    callbacks.extend([es])

    try:
        for maxlen in range(4, 200, 5):
            # val_generator = GzipToNextStepGenerator(val_gzip, grammar_filepath, batch_size, maxSentenceLen=maxlen)
            # generator = MockNextStepGenerator(batch_size=batch_size, maxlen=maxlen)
            # generator = MockDataGenerator(batch_size=batch_size, maxlen=maxlen)

            generators_params = {}
            generators_params.update(
                grammar_filepath=grammar_filepath,
                batch_size=batch_size,
                maxlen=maxlen,
                nb_lines=nb_lines)
            train_generator = GzipToNextToken_KerasGenerator(gzip_filepath=train_gzip, **generators_params)
            val_generator = GzipToNextToken_KerasGenerator(gzip_filepath=val_gzip, **generators_params)

            LM.fit_generator(
                generator=train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                use_multiprocessing=False,
                workers=1)
            LM.save(LM_path)

    except KeyboardInterrupt:
        print("Training interrupted by the user")
        LM.save(LM_path)

    return LM


def with_transformer(
        train_gzip,
        val_gzip,
        grammar_filepath,
        batch_size,
        vocab_size,
        emb_dim,
        units,
        epochs,
        nb_lines,
        LM_path,
        log_path):
    maxlen = 200
    steps_per_epoch = int(nb_lines / batch_size)
    train_generator = generateFromGzip(train_gzip, batch_size)
    val_generator = generateFromGzip(val_gzip, batch_size)

    t_object = TransformerTraining(grammar_filepath=grammar_filepath, maxlen=maxlen, latentDim=units)
    t_object.train(
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        modelFilename='somewhere', verbose=1)


    indices = next(val_generator)
    for frase in indices:
        print('')
        input_seq = frase[:-2]
        softmax = t_object.s2s.next_symbol_prediction(input_seq)
        print(softmax)
    # LM.save(LM_path)
    LM = 0
    return LM


class TransformerTraining(object):

    def __init__(self, grammar_filepath, maxlen, latentDim=16):

        # Transformer definitions
        grammar = nltk.data.load('file:' + grammar_filepath)
        generator_class = dd.sentencesFromGrammar_generator(grammar)
        itokens = generator_class.itokens

        self.s2s = Transformer(itokens, itokens, len_limit=maxlen, d_model=latentDim,
                               d_inner_hid=512,
                               n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)
        self.s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
        self.s2s.output_model.summary()

    def _generate_training_data(self, generator):

        while True:
            sentences = next(generator)
            indices = self.s2s.sentences2indices(sentences)

            input_indices = indices[:, :-1]
            output_indices = indices[:, 1:]

            yield [input_indices, output_indices], None

    def train(self, train_generator, val_generator,
              epochs,
              steps_per_epoch=32,
              batch_size=32,
              modelFilename=None, verbose=1):

        callbacks = []

        try:
            train_generator = self._generate_training_data(train_generator)
            val_generator = self._generate_training_data(val_generator)
            self.s2s.model.fit_generator(train_generator,
                                         epochs=epochs,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_data=val_generator,
                                         validation_steps=2,
                                         use_multiprocessing=False,
                                         shuffle=False,
                                         verbose=verbose,
                                         callbacks=callbacks)

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user")

        self.s2s.output_model.save_weights(modelFilename)

    def getLanguageModel(self):
        return self.s2s.output_model


if __name__ == '__main__':
    gzipDatasetFilepath = '../data/REBER_biased_train.gz'
    sort_gzip_by_length(gzipDatasetFilepath)
