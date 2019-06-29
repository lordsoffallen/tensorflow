from tensorflow.python.keras.callbacks import EarlyStopping
from util import models, vectorize

# Limit on the number of features.
TOP_K = 20000


def train_ngram(train, val, epochs=1000, batch_size=128, layers=2, units=64, dropout_rate=0.2):
    """ Trains n-gram model on the given dataset.

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
    layers: int
        Number of `Dense` layers in the model.
    units: int
        Output dimension of Dense layers in the model.
    dropout_rate: float
        Percentage of input to drop at Dropout layers.

    Returns
    -------
    acc, loss: (float, float)
        Return validation acc and loss
    """

    (train_texts, train_labels) = train
    (val_texts, val_labels) = val

    # Vectorize texts.
    X_train, X_val = vectorize.ngram(train_texts, train_labels, val_texts)

    # Create model instance.
    model = models.MLP(layers=layers, units=units,
                       dropout_rate=dropout_rate,
                       input_shape=X_train.shape[1:])

    stop = EarlyStopping(monitor='val_loss', patience=2)

    # Train and validate model.
    history = model.fit(X_train, train_labels,
                        epochs=epochs,
                        callbacks=[stop],
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    # Save model.
    model.save('mlp_model.h5')
    return history.history['val_acc'][-1], history.history['val_loss'][-1]


def train_sequence(train, val, epochs=1000, batch_size=128, blocks=2, filters=64,
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

    # Create model instance.
    model = models.sepCNN(blocks=blocks, filters=filters,
                          kernel_size=kernel_size,
                          embedding_dim=embedding_dim,
                          dropout_rate=dropout_rate,
                          pool_size=pool_size,
                          input_shape=X_train.shape[1:],
                          num_features=num_features)

    stop = EarlyStopping(monitor='val_loss', patience=2)

    # Train and validate model.
    history = model.fit(X_train, train_labels,
                        epochs=epochs,
                        callbacks=[stop],
                        validation_data=(X_val, val_labels),
                        batch_size=batch_size)

    # Save model.
    model.save('sequence_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]
