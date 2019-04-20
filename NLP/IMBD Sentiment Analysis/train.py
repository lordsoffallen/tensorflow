import argparse
import time
import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import util

FLAGS = None

# Limit on the number of features.
TOP_K = 20000

def ngram(train, val, learning_rate=1e-3, epochs=1000, batch_size=128, 
          layers=2, units=64, dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    Args:
        train: A tuple for training (features, labels)
        val: A tuple for validation (features, labels)
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float, percentage of input to drop at Dropout layers.
    """

    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val = util.vectorize.ngram_vectorize(train_texts, train_labels, val_texts)

    # Create model instance.
    model = util.build_model.MLP(layers=layers, units=units,
                                 dropout_rate=dropout_rate,
                                 input_shape=X_train.shape[1:])

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(X_train, train_labels, epochs=epochs,
                        callbacks=callbacks, verbose=2,
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

def sequence(train, val, learning_rate=1e-3, epochs=1000, batch_size=128,
             blocks=2, filters=64, dropout_rate=0.2, embedding_dim=200,
             kernel_size=3, pool_size=3):
    """Trains sequence model on the given dataset.

    Args:
        train: A tuple for training (features, labels)
        val: A tuple for validation (features, labels)
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
    """

    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val, word_index = util.vectorize.sequence_vectorize(train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = util.build_model.sepCNN(blocks=blocks, filters=filters,
                                    kernel_size=kernel_size, 
                                    embedding_dim=embedding_dim,
                                    dropout_rate=dropout_rate,
                                    pool_size=pool_size,
                                    input_shape=X_train.shape[1:],
                                    num_features=num_features)

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(X_train, train_labels,
                        epochs=epochs, verbose=2,
                        callbacks=callbacks,
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

def tune_ngram(train, val):
    """Tunes n-gram model on the given dataset.

    Args:
        train: A tuple for training (features, labels)
        val: A tuple for validation (features, labels)
    """
    
    # Select parameter values to try.
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]

    # Save parameter combination and results.
    params = {'layers': [], 'units': [], 'accuracy': []}

    # Iterate over all parameter combinations.
    for layers in num_layers:
        for units in num_units:
                params['layers'].append(layers)
                params['units'].append(units)
                accuracy, _ = ngram(train, val, layers=layers, units=units)
                print(('Accuracy: {accuracy}, Parameters: (layers={layers}, '
                       'units={units})').format(accuracy=accuracy,
                                                layers=layers, units=units))
                params['accuracy'].append(accuracy)

    plot_params(params)

def plot_params(params):
    """Creates a 3D surface plot of given parameters.

    Args:
        params: dict, contains layers, units and accuracy value combinations.
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(params['layers'],
                    params['units'],
                    params['accuracy'],
                    cmap=cm.coolwarm,
                    antialiased=False)
    plt.show()

def get_embedding(word_index, embedding_dir, embedding_dim):
    """Gets embedding matrix from the embedding index data.

    Args:
        word_index: dict, word to index map that was generated from the data.
        embedding_dir: string, path to the pre-training embeddings.
        embedding_dim: int, dimension of the embedding vectors.

    Returns:
        dict, word vectors for words in word_index from pre-trained embedding.

    References:
        https://nlp.stanford.edu/projects/glove/

        Download and uncompress archive:
        http://nlp.stanford.edu/data/glove.6B.zip
    """

    # Read the pre-trained embedding file and get word to word vector mappings.
    embedding_matrix_all = {}

    # We are using 200d GloVe embeddings.
    fname = os.path.join(embedding_dir, 'glove.6B.200d.txt')
    with open(fname) as f:
        for line in f:  # Every line contains word followed by the vector value
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix_all[word] = coefs

    # Prepare embedding matrix with just the words in our word_index dictionary
    num_words = min(len(word_index) + 1, TOP_K)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= TOP_K:
            continue
        embedding_vector = embedding_matrix_all.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def fine_tuned_sequence(train, val, embedding_dir, learning_rate=1e-3,
                        epochs=1000, batch_size=128, blocks=2,
                        filters=64, dropout_rate=0.2, embedding_dim=200,
                        kernel_size=3, pool_size=3):
    """Trains sequence model on the given dataset.

    Args:
        train: A tuple for training (features, labels)
        val: A tuple for validation (features, labels)
        embedding_dir: string, path to the pre-training embeddings.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
    """
    
    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val, word_index = util.vectorize.sequence_vectorize(train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    embedding_matrix = get_embedding(word_index, embedding_dir, embedding_dim)

    # Create model instance. First time we will train rest of network while
    # keeping embedding layer weights frozen. So, we set
    # is_embedding_trainable as False.
    model = util.build_model.sepCNN(blocks=blocks, filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=X_train.shape[1:],
                                     num_features=num_features,
                                     pretrained_embedding=True,
                                     embedding_trainable=False,
                                     embedding_matrix=embedding_matrix)

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    model.fit(X_train, train_labels, epochs=epochs,
              callbacks=callbacks, verbose=2,
              validation_data=(X_val, val_labels),
              batch_size=batch_size)

    # Save the model.
    model.save_weights('sequence_model_with_pre_trained_embedding.h5')

    # Create another model instance. This time we will unfreeze the embedding
    # layer and let it fine-tune to the given dataset.
    model = util.build_model.sepCNN(blocks=blocks, filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=X_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features,
                                     pretrained_embedding=True,
                                     embedding_trainable=True,
                                     embedding_matrix=embedding_matrix)

    # Compile model with learning parameters.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Load the weights that we had saved into this new model.
    model.load_weights('sequence_model_with_pre_trained_embedding.h5')

    # Train and validate model.
    history = model.fit(X_train, train_labels,
                        epochs=epochs, verbose=2,
                        callbacks=callbacks,
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('sepcnn_fine_tuned_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

def data_generator(X, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.

    Args:
        X: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.

    Returns:
        Yields feature and label data in batches.
    """

    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    # generate indefinitely
    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield X_batch, y_batch

def batch_sequence(train, val, learning_rate=1e-3, epochs=1000, 
                   batch_size=128, blocks=2, filters=64, dropout_rate=0.2,
                   embedding_dim=200, kernel_size=3, pool_size=3):
    """Trains sequence model on the given dataset.

    Args:
        train: A tuple for training (features, labels)
        val: A tuple for validation (features, labels)
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
    """
     
    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val, word_index = util.vectorize.sequence_vectorize(train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = util.build_model.sepCNN(blocks=blocks, filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=X_train.shape[1:],
                                     num_features=num_features)

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Create training and validation generators.
    train_generator = data_generator(X_train, train_labels, num_features, batch_size)
    val_generator = data_generator(X_val, val_labels, num_features, batch_size)

    # Get number of training steps. This indicated the number of steps it takes
    # to cover all samples in one epoch.
    steps_per_epoch = X_train.shape[0] // batch_size
    if X_train.shape[0] % batch_size:
        steps_per_epoch += 1

    # Get number of validation steps.
    val_steps = X_val.shape[0] // batch_size
    if X_val.shape[0] % batch_size:
        val_steps += 1

    # Train and validate model.
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  epochs=epochs, verbose=2)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('batch_sequence_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='if you do not have the data locally \
                        (at "~/.keras/datasets/" + path), it will be \
                        downloaded to provided location.')

    parser.add_argument('--model', type=str, default='ngram',
                        help='Which model to use for training')

    parser.add_argument('--learning_rate', type=int, default=1e-3,
                        help='Model learning rate parameter')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='How many passes to make over the dataset before\
                            training stops')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Amount of data to train per epochs')

    parser.add_argument('--embedding_dir', type=str, default='./data',
                        help='embedding input data directory')

    #TODO Add more parser arguments

    FLAGS, unparsed = parser.parse_known_args()

    train, val = imdb.load_data(path="imdb.npz", num_words=None,
                                skip_top=0, maxlen=None, seed=113,
                                start_char=1, oov_char=2, index_from=3)
    
    if FLAGS.model == 'ngram':
        ngram(train, val, FLAGS.learning_rate, FLAGS.epochs, FLAGS.batch_size)
    else:
        sequence(train, val, FLAGS.learning_rate, FLAGS.epochs, FLAGS.batch_size)
    

