from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io


VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRUNC = 'post'
PADDING = 'post'
OOV = "<OOV>"
EPOCHS = 30
RATIO = .8
STOPWORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my",
    "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
    "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
]


def plot_graphs(history, sel):
    """Plots the training graph from Keras History object"""
    plt.plot(history.history[sel])
    plt.plot(history.history['val_' + sel])
    plt.xlabel("Epochs")
    plt.ylabel(sel)
    plt.legend([sel, 'val_' + sel])
    plt.show()


def read_data():
    """ Reads bbc text file from a csv file. Returns sentences and labels """
    df = pd.read_csv('bbc-text.csv')
    sentences = df.text.to_list()
    labels = df.category.to_list()
    return sentences, labels


def process(sentence):
    """ Applies word tokenizing to input sentence and removes the stopwords. """
    tokenized_words = text_to_word_sequence(sentence)
    words = [word for word in tokenized_words if word not in STOPWORDS]
    return ' '.join(words)


def split_data(sentences, labels, flush=True):
    """ Splits data into train and validation part.

    Parameters
    ----------
    sentences: list
        A list of sentences
    labels: list
        A list of labels
    flush: bool
        Print the length of inputs

    Returns
    -------
    train_sentences, train_labels, val_sentences, val_labels:
        Returns train and validation data

    """

    train_size = int(len(sentences) * RATIO)

    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:]
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]

    if flush:
        print('Length of train data : ', train_size, '....')
        print('Length of train sentences : ', len(train_sentences), '....')
        print('Length of train labels : ', len(train_labels), '....')
        print('Length of validation sentences : ', len(val_sentences), '....')
        print('Length of validation labels : ', len(val_labels), '....')

    return train_sentences, train_labels, val_sentences, val_labels


def convert_seq(train_sentences, train_labels, val_sentences, val_labels, labels):
    """ Converts sentences to sequences for model training

    Parameters
    ----------
    train_sentences: list of str
        A list of sentences
    train_labels: list of str
        A list of train labels
    val_sentences: list of str
        A list of validation sentences
    val_labels: list of str
        A list of validation labels
    labels: list of str
        A list of complete labels to fit on tokenizer

    Returns
    -------
    word_index, train_padded, train_label_seq, val_padded, val_label_seq:
        Returns train and validation sequences
    """

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=PADDING, maxlen=MAX_LENGTH)
    print('Example Sentence')
    print('{} \n---------- is mapped to this padded sequence :\n{}'.format(train_sentences[1], train_padded[1]))

    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded = pad_sequences(val_sequences, padding=PADDING, maxlen=MAX_LENGTH)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    val_label_seq = np.array(label_tokenizer.texts_to_sequences(val_labels))

    return word_index, train_padded, train_label_seq, val_padded, val_label_seq


def build_model(summary=True):
    """Builds the Keras model and returns it. If summary is true, print model summary."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if summary:
        model.summary()

    return model


def reverse_word_index(word_index):
    """Return a reversed word index from word index dictionary"""
    return dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text, word_index):
    """Decode integer sequence to text format."""
    reversed_word_index = reverse_word_index(word_index)
    return ' '.join([reversed_word_index.get(i, '?') for i in text])


def export_tsv(model, word_index):
    """ Export model embeddings and meta information to a tsv files.
    This file can be used in visualization at https://projector.tensorflow.org

    Parameters
    ----------
    model: tf.keras.Model
        Trained keras model
    word_index: dict
        A dict contains word indexes
    """

    e = model.layers[0]
    weights = e.get_weights()[0]
    print('shape: (vocab_size, embedding_dim): ', weights.shape)

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    reversed_word_index = reverse_word_index(word_index)

    for word_num in range(1, VOCAB_SIZE):
        word = reversed_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

    out_v.close()
    out_m.close()


if __name__ == '__main__':
    sentences, labels = read_data()
    print('Data Labels: ', set(labels))
    sentences = [process(sentence) for sentence in sentences]
    print('Length of Labels : ', len(labels), '....')
    print('Length of Sentences : ', len(sentences), '....')

    train_sentences, train_labels, val_sentences, val_labels = split_data(sentences, labels)
    word_index, train_padded, train_label_seq, val_padded, val_label_seq = convert_seq(train_sentences,
                                                                                       train_labels,
                                                                                       val_sentences,
                                                                                       val_labels,
                                                                                       labels)

    model = build_model()
    history = model.fit(train_padded,
                        train_label_seq,
                        epochs=EPOCHS,
                        validation_data=(val_padded, val_label_seq),
                        verbose=2)

    plot_graphs(history, "acc")
    plot_graphs(history, "loss")

    export_tsv(model, word_index)

    print('Go to https://projector.tensorflow.org and upload the generated '
          'tsv files there to see embeddings visualization')




