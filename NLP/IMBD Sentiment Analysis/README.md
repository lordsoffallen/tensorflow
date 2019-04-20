# Text Classification on IMBD Dataset

1.  *util/explore* - Helper functions to understand datasets.

2.  *util/vectorize* - N-gram and sequence vectorization functions.

3.  *util/build_model* - Helper functions to create multi-layer perceptron and
    separable convnet models.

4.  *train* - Demonstrates how to use all of the above modules and train a
    model.

    + *ngram* - Trains a multi-layer perceptron model.

    + *sequence* - Trains a sepCNN model.

    + *fine_tuned_sequence* - Trains a sepCNN model with
    pre-trained embeddings that are fine-tuned.

    + *batch_sequence* - Same as *sequence* but here
    we are training data in batches.

    + *tune_ngram* - Tune MLP model parameters.
