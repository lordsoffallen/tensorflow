# Text Classification Guide with IMDB

This repository contains end-to-end text classification guide to solve
text classification problems using machine learning.


## Scripts

2.  `util/explore.py` - Helper functions to plot datasets.

3.  `util/vectorize.py` - N-gram and sequence vectorization functions.

4.  `util/models.py` - Functions to create multi-layer perceptron and
    separable convnet models.

5.  `train.py` - Functions to train sequence or ngram models.

    + *train_ngram()* - Trains a multi-layer perceptron model.

    + *train_sequence()* - Trains a sepCNN model.

6.  `fine_tune.py` - Script to find the best hyper-parameter values for the model.

    + *fine_tune_sequence()* - Trains a sepCNN model with
    pre-trained embeddings that are fine-tuned

    + *fine_tune_ngram()* - Trains a MLP model with
    different model parameters.