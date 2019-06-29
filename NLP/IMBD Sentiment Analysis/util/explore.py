from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_num_words(sample):
    """ Gets the median number of words per sample given corpus.

    Parameters
    ----------
    sample: list of str
        A list containing sample texts.

    Returns
    -------
    num_words: int
        Integer, median number of words per sample.
    """

    return np.median([len(s.split()) for s in sample])


def plot_freq_dist(sample, ngram_range=(1, 2), max_ngrams=50):
    """Plots the frequency distribution given a ngram range.

    Parameters
    ----------
    sample: list of str
        A list containing sample texts.
    ngram_range: tuple
        The range of n-gram values to consider.
    max_ngrams: int
        number of n-grams to plot.
    """

    vectorizer = CountVectorizer(analyzer='word', dtype='int32',
                                 ngram_range=ngram_range, strip_accents='unicode',
                                 decode_error='replace', max_features=max_ngrams)
    vectorized_texts = vectorizer.fit_transform(sample)
    ngrams = list(vectorizer.get_feature_names())

    # Add up the counts column-wise per n-gram
    counts = vectorized_texts.sum(axis=0).tolist()[0]
    
     # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    counts, ngrams = zip(*[(c, n) for c, n in sorted(zip(counts, ngrams), reverse=True)])
    ngrams = list(ngrams)[:max_ngrams]
    counts = list(counts)[:max_ngrams]

    idx = np.arange(max_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=90)
    plt.show()


def plot_length_dist(sample):
    """Plots the sample length distribution.

    Parameters
    ----------
    sample: list of str
        A list containing sample texts.
    """

    plt.hist([len(s) for s in sample], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_class_dist(labels):
    """Plots the class distribution.

    Parameters
    ----------
    labels: list
        A list containing label values.
    """

    num_classes = 2
    count_map = Counter(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class distribution')
    plt.xticks(idx, idx)
    plt.show()
