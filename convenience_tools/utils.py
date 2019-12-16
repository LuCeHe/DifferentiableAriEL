import gzip
from tqdm import tqdm

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



if __name__ == '__main__':
    gzipDatasetFilepath = '../data/REBER_biased_train.gz'
    sort_gzip_by_length(gzipDatasetFilepath)
