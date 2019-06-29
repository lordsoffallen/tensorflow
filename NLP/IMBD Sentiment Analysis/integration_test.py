""" Module to test training accuracy. We will measure the accuracies
at the end and check that they are within +/- 2% of an expected number. """

from tensorflow.python.keras.datasets import imdb
import pytest
import train as training
import fine_tune as tune


def test_train_ngram():
    train, val = imdb.load_data()
    acc, loss = training.train_ngram(train, val)
    assert acc == pytest.approx(0.91, 0.02)
    assert loss == pytest.approx(0.24, 0.02)


def test_train_sequence():
    train, val = imdb.load_data()
    acc, loss = training.train_sequence(train, val)
    assert acc == pytest.approx(0.68, 0.02)
    assert loss == pytest.approx(0.82, 0.02)


def test_fine_tuned_sequence():
    train, val = imdb.load_data()
    acc, loss = tune.fine_tune_sequence(train, val)
    assert acc == pytest.approx(0.84, 0.02)
    assert loss == pytest.approx(0.55, 0.02)


def test_fine_tune_ngram():
    train, val = imdb.load_data()
    acc, loss = tune.fine_tune_ngram(train, val)
    assert acc == pytest.approx(0.61, 0.02)
    assert loss == pytest.approx(0.89, 0.02)
