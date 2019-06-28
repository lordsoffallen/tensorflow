from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical, get_file
from tensorflow.python.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.python.keras.models import Model, save_model
from tensorflow.python.keras.initializers import Constant
from sklearn.model_selection import train_test_split
import os
import numpy as np


GLOVE_DIR = './keras/models/glove.6B.100d.txt'
DATA_URL = 'http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def read_glove(data_dir):
    """ Builds index mapping words in the embeddings set to their embedding vector

    Parameters
    ----------
    data_dir: str
        Directory where the GloVe embeddings are located

    Returns
    -------
    word2embed: dict
        A dict contains word to embeddings mapping
    """

    def process(l):
        """Returns word, coeffs couple from a line"""
        values = l.split()
        return values[0], np.asarray(values[1:], dtype='float32')

    print('Reading GloVe Word Vectors....')
    with open(data_dir) as f:
        word2embed = dict(process(line) for line in f)

    print('Found {} Word Vectors.'.format(len(word2embed)))
    return word2embed


def build_dataset():
    """Reads 20 new dataset and returns text(list), labels(list) and a dict of labels to index """
    print('Processing Dataset...')
    zipped = get_file('20news-18828.tar.gz', DATA_URL, extract=True)
    data_dir = zipped[:-7]  # remove .tar.gz

    # Text samples and their labels
    texts = []  # list of text samples
    label2idx = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    for folder in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, folder)
        if os.path.isdir(path):
            label = len(label2idx)
            label2idx[folder] = label

            for fname in sorted(os.listdir(path)):
                if fname.isdigit():    # A text file
                    fpath = os.path.join(path, fname)

                    with open(fpath, encoding='latin-1') as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i: t = t[i:]
                        texts.append(t)
                    labels.append(label)

    print('Found {} texts.' .format(len(texts)))

    return texts, labels, label2idx


def vectorize(texts, labels, num_words, maxlen):
    """ Vectorize the text samples into sequences.

    Parameters
    ----------
    texts: list of str
        A list of text data
    labels: list of int
        A list of labels integer encoded
    num_words: int
        Maximum number of words
    maxlen: int
        Maximum sequence length

    Returns
    -------
    X, y, word_index:
        Returns data, labels and word index
    """

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found {} unique tokens...'.format(len(word_index)))

    X = pad_sequences(sequences, maxlen=maxlen)
    y = to_categorical(np.asarray(labels))

    print('Features len: {}  Labels len: {}'.format(X.shape, y.shape))
    return X, y, word_index


def get_embeddings(word2embed, dim, max_num_words, word_index):
    """ Retrieves embedding weights

    Parameters
    ----------
    word2embed: dict
        Dict mapping word to embeddings
    dim: int
        Embedding Dimension size
    max_num_words: int
        Maximum number of words
    word_index: dict
        Word to index mapping

    Returns
    -------
    embedding:
        A numpy array containing embedding weights
    """

    num_words = min(max_num_words, len(word_index)) + 1
    embedding = np.zeros((num_words, dim))
    for word, idx in word_index.items():
        if idx > max_num_words: continue
        embedding[idx] = word2embed.get(word, '0.0')

    return embedding


def build_model(embedding_weights, embedding_dim, num_words, input_length, num_classes=20):
    """ Builds a Keras model. It sets embeddings layer trainable to False
    to keep the embeddings fixed

    Parameters
    ----------
    embedding_weights: np.ndarray
        A numpy array contains embedding weights
    embedding_dim: int
        Embeddings dimension
    num_words: int
        Number of words in the dataset
    input_length: int
        Maximum sequence length
    num_classes: int
        Number of classes in the dataset

    Returns
    -------
    model: Model
        A keras compiled model instance
    """

    embedding = Embedding(num_words,
                          embedding_dim,
                          embeddings_initializer=Constant(embedding_weights),
                          input_length=input_length,
                          trainable=False)

    seq_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding(seq_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(seq_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model


if __name__ == '__main__':
    word2embed = read_glove(GLOVE_DIR)
    texts, labels, label2idx = build_dataset()
    X, y, wordidx = vectorize(texts, labels, num_words=MAX_NUM_WORDS, maxlen=MAX_SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

    embedding_weights = get_embeddings(word2embed, dim=EMBEDDING_DIM, max_num_words=MAX_NUM_WORDS, word_index=wordidx)
    num_words = min(MAX_NUM_WORDS, len(wordidx)) + 1
    model = build_model(embedding_weights,
                        embedding_dim=EMBEDDING_DIM,
                        num_words=num_words,
                        input_length=MAX_SEQUENCE_LENGTH,
                        num_classes=len(label2idx))
    model.summary()
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))
    print('Saving model...')
    save_model(model, '20_news.h5')
