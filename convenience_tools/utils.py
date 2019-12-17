import gzip

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tqdm import tqdm

from DifferentiableAriEL.nnets.tf_tools.keras_layers import predefined_model
from GenericTools.LeanguageTreatmentTools.sentence_generators import GzipToNextStepGenerator, \
    MockNextStepGenerator, MockDataGenerator, GzipToNextToken_KerasGenerator


def sort_gzip_by_length(gzipDatasetFilepath):
    f = gzip.open(gzipDatasetFilepath, 'rb')

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
        gzip_filepath,
        grammar_filepath,
        batch_size,
        vocab_size,
        emb_dim,
        epochs,
        nb_lines,
        steps_per_epoch,
        LM_path,
        log_path):

    LM = predefined_model(vocab_size, emb_dim)
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
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    callbacks.extend([tb, es])
    """
    try:
        for maxlen in range(4, 200, 5):
            old_generator = GzipToNextStepGenerator(gzip_filepath, grammar_filepath, batch_size, maxSentenceLen=maxlen)
            #generator = MockNextStepGenerator(batch_size=batch_size, maxlen=maxlen)
            #generator = MockDataGenerator(batch_size=batch_size, maxlen=maxlen)
            generator = GzipToNextToken_KerasGenerator(
                gzip_filepath=gzip_filepath,
                grammar_filepath=grammar_filepath,
                batch_size=batch_size,
                maxlen=maxlen,
                nb_lines=nb_lines)
            validation_data = next(old_generator)

            LM.fit_generator(
                generator,
                epochs=epochs,
                validation_data=validation_data,
                use_multiprocessing=False,
                #steps_per_epoch=steps_per_epoch,
                workers=1)
            LM.save(LM_path)

    except KeyboardInterrupt:
        print("Training interrupted by the user")
        LM.save(LM_path)

    return LM


if __name__ == '__main__':
    gzipDatasetFilepath = '../data/REBER_biased_train.gz'
    sort_gzip_by_length(gzipDatasetFilepath)


