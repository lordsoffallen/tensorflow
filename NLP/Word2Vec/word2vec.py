from tensorflow.python.keras.layers import Input, Dense, Reshape, Embedding
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import get_file
from zipfile import ZipFile
from collections import Counter
import numpy as np
import tensorflow as tf


EPOCHS = 200000
VOCAB_SIZE = 10000      # Total number of different words in the vocabulary
EMBEDDING_DIM = 300     # Dimension of the embedding vector
MIN_OCCURANCE = 10      # Remove all words that does not appears at least n times
WINDOW_SIZE = 3         # How many words to consider left and right


class SimilarityCallback:
    def __init__(self, word2idx, val_model, top_k=8, vocab_size=10000, valid_size=16, valid_window=100):
        """ Initialize the cosine similarity callback class.

        Parameters
        ----------
        word2idx: dict
            A dict contains word to int mappings
        val_model: tf.keras.Model
            A keras validation model to check similarity results
        top_k: int
            Number of nearest neighbors
        vocab_size: int
            Number of words in the vocabulary
        valid_size: int
            Random set of words to evaluate similarity on.
        valid_window: int
            Only pick dev samples in the head of the distribution.
        """
        self.idx2word = reverse_dict(word2idx)
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.val_model = val_model

    def run(self):
        """Runs similarity on validation model and prints the nearest results"""
        for i in range(self.valid_size):
            valid_word = self.idx2word[self.valid_examples[i]]
            sim = self._get_sim(self.valid_examples[i])
            nearest = (-sim).argsort()[1:self.top_k + 1]
            log_str = 'Nearest to {}:'.format(valid_word)
            close_words = [self.idx2word[nearest[k]] for k in range(self.top_k)]
            print(log_str, close_words)

    def _get_sim(self, valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        arr1 = np.zeros((1,))
        arr2 = np.zeros((1,))
        arr1[0,] = valid_word_idx
        for i in range(self.vocab_size):
            arr2[0,] = i
            out = self.val_model.predict_on_batch([arr1, arr2])
            sim[i] = out

        return sim


def reverse_dict(d):
    return dict(zip(d.values(), d.keys()))


def build_dataset(max_words=10000):
    """ It will download the file from http://mattmahoney.net/dc/text8.zip
    if it doesnt exist in ./keras/datasets/ folder. Then loads the data
    process by the common words to construct the vocabulary

    Parameters
    ----------
    max_words: int
        Maximum Number of words to include in our vocabulary

    Returns
    -------
    data, word2idx:
        Contains text data and words to int dict mapping
    """

    def read_data(fname):
        """Extract the first file enclosed in a zip file as a list of words."""
        with ZipFile(fname) as f:
            text_words = f.read(f.namelist()[0]).lower().split()
        return text_words

    filename = get_file('text8.zip', 'http://mattmahoney.net/dc/text8.zip')
    words = read_data(filename)

    # Build the dictionary and replace rare words with UNK token
    count = [['UNK', -1]]

    # Retrieve the most common words
    count.extend(Counter(words).most_common(max_words - 1))

    # Remove samples with less than 'min_occurrence' occurrences
    for i in range(len(count) - 1, -1, -1):
        if count[i][1] < MIN_OCCURANCE:
            count.pop(i)
        else:
            # The collection is ordered, so stop when 'MIN_OCCURANCE' is reached
            break

    # Compute the vocabulary size
    vocabulary_size = len(count)

    # Assign an id to each word
    word2idx = dict()
    for i, (word, _) in enumerate(count):
        word2idx[word] = i

    data = list()
    unk_count = 0
    for word in words:
        # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
        index = word2idx.get(word, 0)
        if index == 0:
            unk_count += 1

        data.append(index)

    count[0] = ('UNK', unk_count)

    print("Words count: ", len(words))
    print("Vocabulary size: ", vocabulary_size)
    print("Most common words: ", count[:10])

    return data, word2idx


def build_model():
    """ Builds word2vec model and return training and validation models."""

    embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=1, name='embedding')

    target_inputs = Input((1,))
    context_inputs = Input((1,))

    target = embedding(target_inputs)
    target = Reshape((EMBEDDING_DIM, 1))(target)

    context = embedding(context_inputs)
    context = Reshape((EMBEDDING_DIM, 1))(context)

    # setup a cosine similarity operation which will be output in a secondary model
    similarity = Dot(axes=0, normalize=True)([target, context])

    # now perform the dot product operation to get a similarity measure
    dot_product = Dot(axes=1)([target, context])
    dot_product = Reshape((1,))(dot_product)

    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)

    # create the primary training model
    model = Model([target_inputs, context_inputs], output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # create a secondary validation model to run our similarity checks during training
    val_model = Model([target_inputs, context_inputs], similarity)

    return model, val_model


if __name__ == '__main__':
    data, word2idx = build_dataset(max_words=VOCAB_SIZE)
    print('Printing first 10 words from vocabulary: ', data[:10])

    model, val_model = build_model()

    callback = SimilarityCallback(word2idx, val_model, vocab_size=VOCAB_SIZE)

    sampling_table = sequence.make_sampling_table(VOCAB_SIZE)
    couples, labels = sequence.skipgrams(data, VOCAB_SIZE, window_size=WINDOW_SIZE, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")

    x1,x2, y = np.zeros((1,)), np.zeros((1,)), np.zeros((1,))

    for c in range(EPOCHS):
        idx = np.random.randint(0, len(labels)-1)
        x1[0, ] = word_target[idx]
        x2[0, ] = word_context[idx]
        y[0, ] = labels[idx]    # As an np.array
        loss = model.train_on_batch([x1, x2], y)
        if c % 100 == 0:
            print("Iteration: {}, Loss: {}".format(c, loss))
        if c % 10000 == 0:
            callback.run()

    tf.keras.models.save_model(model, 'word2vec.h5')
    print('Model saved...')
