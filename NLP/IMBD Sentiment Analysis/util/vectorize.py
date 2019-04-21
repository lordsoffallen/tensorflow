import tensorflow as tf
import numpy as np

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Vectorization parameters
NGRAM_RANGE = (1, 2)
TOKEN_MODE = 'word'
MAX_SEQUENCE_LENGTH = 500

# Limit on the number of features.
TOP_K = 20000

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    Args:
        train_texts: A list containing training text strings.
        train_labels: A ndarray containing training labels.
        val_texts: A list containing validation text strings.

    Returns:
        train, val: A numpy array vectorized training and validation texts
    """
    
    vectorizer = TfidfVectorizer(analyzer=TOKEN_MODE, ngram_range=NGRAM_RANGE,
                                 min_df=MIN_DOCUMENT_FREQUENCY, strip_accents='unicode',
                                 decode_error='replace', dtype=np.float32)
    train = vectorizer.fit_transform(train_texts)
    val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, train.shape[1]))
    selector.fit(train, train_labels)
    train = selector.transform(train)
    val = selector.transform(val)

    return train, val


def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.
    1 text = 1 sequence vector with fixed length.

    Args:
        train_texts: A list, training text strings.
        val_texts: A list, validation text strings.

    Returns:
        train, val, word_index: vectorized training and validation
                                texts and word index dictionary.
    """

    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    train = tokenizer.texts_to_sequences(train_texts)
    val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    train = sequence.pad_sequences(train, maxlen=max_length)
    val = sequence.pad_sequences(val, maxlen=max_length)
    return train, val, tokenizer.word_index
