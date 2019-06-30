from tensorflow.python.keras.callbacks import EarlyStopping
from util import models, vectorize
import matplotlib.pyplot as plt
import numpy as np
import train as training


GLOVE_DIR = './keras/models/glove.6B.200d.txt'
TOP_K = 20000       # Limit on the number of features.


def read_glove(data_dir):
    """ Builds index mapping words in the embeddings set to their embedding vector

    Parameters
    ----------
    data_dir: str
        Directory where the GloVe embeddings are located

    Returns
    -------
    word2embed: dict
        A dict contains word to embeddings mapping
    """

    def process(l):
        """Returns word, coeffs couple from a line"""
        values = l.split()
        return values[0], np.asarray(values[1:], dtype='float32')

    print('Reading GloVe Word Vectors....')
    with open(data_dir) as f:
        word2embed = dict(process(line) for line in f)

    print('Found {} Word Vectors.'.format(len(word2embed)))
    return word2embed


def get_embeddings(word2embed, dim, word_index):
    """ Retrieves embedding weights

    Parameters
    ----------
    word2embed: dict
        Dict mapping word to embeddings
    dim: int
        Embedding Dimension size
    word_index: dict
        Word to index mapping

    Returns
    -------
    embedding:
        A numpy array containing embedding weights
    """

    num_words = min(len(word_index) + 1, TOP_K)
    embedding = np.zeros((num_words, dim))
    for word, idx in word_index.items():
        if idx > TOP_K: continue
        embedding[idx] = word2embed.get(word, '0.0')

    return embedding


def fine_tune_ngram(train, val):
    """Tunes n-gram model on the given dataset.

    Parameters
    ----------
    train: tuple
        Training data (features, labels)
    val: tuple
        Validation data (features, labels)
    """

    def plot_params(params):
        """Creates a 3D surface plot of given a dict of parameters."""
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(params['layers'], params['units'], params['accuracy'], antialiased=False)
        plt.set_cmap('coolwarm')
        plt.show()

    # Parameter values to try.
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]

    # Save parameter combination and results.
    params = {'layers': [], 'units': [], 'accuracy': []}

    # Iterate over all parameter combinations.
    for layers in num_layers:
        for units in num_units:
            params['layers'].append(layers)
            params['units'].append(units)
            accuracy, _ = training.train_ngram(train, val, layers=layers, units=units)
            print(('Accuracy: {}, Parameters: (layers={}, '
                   'units={})').format(accuracy, layers, units))
            params['accuracy'].append(accuracy)

    plot_params(params)


def fine_tune_sequence(train, val, epochs=1000, batch_size=128, blocks=2, filters=64,
                       dropout_rate=0.2, embedding_dim=200, kernel_size=3, pool_size=3):
    """Trains sequence model on the given dataset.

    Parameters
    ----------
    train: tuple
        Training data (features, labels)
    val: tuple
        Validation data (features, labels)
    epochs: int
        Number of epochs.
    batch_size: int
        Number of samples per batch.
    blocks: int
        Number of pairs of sepCNN and pooling blocks in the model.
    filters: int
        Output dimension of sepCNN layers in the model.
    dropout_rate: float
        Percentage of input to drop at Dropout layers.
    embedding_dim: int
        Dimension of the embedding vectors.
    kernel_size: int
        Length of the convolution window.
    pool_size: int
        Factor by which to downscale input at MaxPooling layer.

    Returns
    -------
    acc, loss: (float, float)
        Return validation acc and loss
    """

    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val, word_index = vectorize.sequence(train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    word2embed = read_glove(GLOVE_DIR)
    embedding_weights = get_embeddings(word2embed, embedding_dim, word_index)

    # First time we will train rest of network while keeping embedding layer
    # weights frozen. So, we set embedding_trainable as False.
    model = models.sepCNN(blocks=blocks,
                          filters=filters,
                          kernel_size=kernel_size,
                          embedding_dim=embedding_dim,
                          dropout_rate=dropout_rate,
                          pool_size=pool_size,
                          input_shape=X_train.shape[1:],
                          num_features=num_features,
                          pretrained_embedding=True,
                          embedding_trainable=False,
                          embedding_weights=embedding_weights)

    stop = EarlyStopping(monitor='val_loss', patience=2)

    # Train and validate model.
    model.fit(X_train, train_labels,
              epochs=epochs,
              callbacks=[stop],
              validation_data=(X_val, val_labels),
              batch_size=batch_size)

    # Save the model.
    model.save_weights('seq_model_with_pre_trained_embedding.h5')

    # Create another model instance. This time we will unfreeze the embedding
    # layer and let it fine-tune to the given dataset.
    model = models.sepCNN(blocks=blocks, filters=filters,
                          kernel_size=kernel_size,
                          embedding_dim=embedding_dim,
                          dropout_rate=dropout_rate,
                          pool_size=pool_size,
                          input_shape=X_train.shape[1:],
                          num_features=num_features,
                          pretrained_embedding=True,
                          embedding_trainable=True,
                          embedding_weights=embedding_weights)

    # Load the weights
    model.load_weights('seq_model_with_pre_trained_embedding.h5')

    history = model.fit(X_train, train_labels,
                        epochs=epochs,
                        callbacks=[stop],
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    model.save('fine_tuned_seq_model.h5')

    return history['val_acc'][-1], history['val_loss'][-1]

