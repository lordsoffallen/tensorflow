from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np


NGRAM_RANGE = (1, 2)
MAX_SEQUENCE_LENGTH = 500
TOP_K = 20000       # Limit on the number of features.
MIN_DOCUMENT_FREQUENCY = 2


def ngram(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors. 1 text = 1 tf-idf vector the length of
    vocabulary of uni-grams + bi-grams. It selects top k(20000 by default) features
    from the text corpus.

    Parameters
    ----------
    train_texts: list of str
        A list containing training text strings.
    train_labels: np.ndarray
        A ndarray containing training labels.
    val_texts: list of str
        A list containing validation text strings.

    Returns
    -------
    train, val: (np.ndarray, np.ndarray)
        A numpy array vectorized training and validation texts
    """
    
    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE,
                                 min_df=MIN_DOCUMENT_FREQUENCY,
                                 strip_accents='unicode',
                                 decode_error='replace',
                                 dtype=np.float32)
    train = vectorizer.fit_transform(train_texts)
    val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, train.shape[1]))
    selector.fit(train, train_labels)
    train = selector.transform(train)
    val = selector.transform(val)

    return train, val


def sequence(train_texts, val_texts):
    """Vectorizes texts as sequence vectors. 1 text = 1 sequence vector with fixed length.
    Also selects the top k(20000 by default) features.

    Parameters
    ----------
    train_texts: list of str
        Training text strings.
    val_texts: list of str
        Validation text strings.

    Returns
    -------
    train, val, word_index:
        Vectorized training and validation texts and word index dictionary.
    """

    # Create vocabulary with training texts.
    tokenizer = Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    train = tokenizer.texts_to_sequences(train_texts)
    val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH: max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value.
    train = pad_sequences(train, maxlen=max_length)
    val = pad_sequences(val, maxlen=max_length)
    return train, val, tokenizer.word_index
