import gzip
import logging

import nltk
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import grammar_on_transformer.dataloader as dd
from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model
from GenericTools.LeanguageTreatmentTools.sentence_generators import GzipToNextToken_KerasGenerator, \
    GzipToIndicesGenerator
from grammar_on_transformer.layers.transformer import Transformer

logger = logging.getLogger(__name__)


def sort_gzip_by_length(gzip_filepath):
    f = gzip.open(gzip_filepath, 'rb')

    sentences = []
    for line in f:
        sentences.append(line)

    sorted_sentences = sorted(sentences, key=len)
    print(sentences)
    print(sorted_sentences)
    destination_path = '../data/sorted_REBER.gz'
    with gzip.open(destination_path, mode='wt') as f:
        for sentence in tqdm(sorted_sentences):
            f.write(sentence.decode("utf-8") + '\r\n')


def train_language_model(
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


def train_language_model_curriculum_learning(
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


def train_language_model_transformer(
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

    maxlen = None
    try:
        generators_params = {}
        generators_params.update(
            grammar_filepath=grammar_filepath,
            batch_size=batch_size,
            maxlen=maxlen,
            nb_lines=nb_lines)

        # Generators
        # train_generator = GzipToNextToken_KerasGenerator(gzip_filepath=train_gzip, **generators_params)
        # val_generator = GzipToNextToken_KerasGenerator(gzip_filepath=val_gzip, **generators_params)

        train_generator = GzipToIndicesGenerator(train_gzip, grammar_filepath, batch_size, maxSentenceLen=maxlen)
        val_generator = GzipToIndicesGenerator(val_gzip, grammar_filepath, batch_size, maxSentenceLen=maxlen)

        LM.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            use_multiprocessing=False,
            workers=1)

    except KeyboardInterrupt:
        print("Training interrupted by the user")

    LM.save(LM_path)

    return LM


class TransformerTraining:
    def __init___(self, grammar_filepath, maxlen, latentDim=16):
        # Transformer definitions
        grammar = nltk.data.load('file:' + grammar_filepath)
        generator_class = dd.sentencesFromGrammar_generator(grammar)
        itokens = generator_class.itokens

        self.s2s = Transformer(itokens, itokens, len_limit=maxlen, d_model=latentDim,
                               d_inner_hid=512,
                               n_head=8, d_k=64, d_v=64, layers=2, dropout=0.1)
        self.s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

    def _generate_training_data(self, generator, batch_size):
        sentenceGenerator = generator(batch_size)
        while True:
            sentences = next(sentenceGenerator)
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
            train_generator = self._generateTrainingData(train_generator, batch_size)
            val_generator = self._generateTrainingData(val_generator, batch_size)
            self.s2s.model.fit_generator(train_generator,
                                         epochs=epochs,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_data=val_generator,
                                         validation_steps=2,
                                         use_multiprocessing=False,
                                         shuffle=False,
                                         verbose=verbose,
                                         callbacks=callbacks)
            self.model.summary()

        except KeyboardInterrupt:
            logger.info("Training interrupted by the user")

        self.model.save_weights(modelFilename)

    def getLanguageModel(self):
        pass


if __name__ == '__main__':
    gzipDatasetFilepath = '../data/REBER_biased_train.gz'
    sort_gzip_by_length(gzipDatasetFilepath)
