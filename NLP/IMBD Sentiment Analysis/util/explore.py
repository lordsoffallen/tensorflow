import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_words(sample_texts):
    """Gets the median number of words per sample given corpus.

    Args:
        sample_texts: A list containing sample texts.

    Returns:
        Integer, median number of words per sample.
    """

    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_freq_dist(sample_texts, ngram_range=(1, 2), max_ngrams=50):
    """Plots the frequency distribution given a ngram range.

    Args:
        sample_texts: A list containing sample texts.
        ngram_range: A tuple (min, mplt), The range of n-gram values to consider.
                    Min and mplt are the lower and upper bound values for the range.
        max_ngrams: int, number of n-grams to plot.
    """

    vectorizer = CountVectorizer(analyzer='word', dtype='int32',
                                 ngram_range=ngram_range, strip_accents='unicode',
                                 decode_error='replace', max_features=max_ngrams)
    vectorized_texts = vectorizer.fit_transform(sample_texts)
    ngrams = list(vectorizer.get_feature_names())

    # Add up the counts per n-gram ie. column-wise
    counts = vectorized_texts.sum(axis=0).tolist()[0]

    idx = np.arange(max_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()


def plot_sample_length_dist(sample_texts):
    """Plots the sample length distribution.

    Args:
        sample_texts: A list containing sample texts.
    """

    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_class_dist(labels):
    """Plots the class distribution.

    Args:
        labels: A list containing label values. There should be at least
                one sample for values in the range (0, num_classes -1)
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
