from tensorflow.python.keras.utils import get_file
import os
import random
import numpy as np

IMDB_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def load_imdb(seed=123):
    """ Loads the Imdb movie reviews sentiment analysis dataset.

    Parameters
    ----------
    seed: int
        Seed for randomizer.

    Returns
    -------
    train_texts, train_labels, test_texts, test_labels:
        A tuple of training and validation data.
    """

    zipped = get_file('aclImdb_v1.tar.gz', IMDB_URL, extract=True)
    data_path = zipped[:-10]  # remove .tar.gz

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return (train_texts, np.array(train_labels)), (test_texts, np.array(test_labels))
