from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, MaxPooling1D
from tensorflow.python.keras.layers import SeparableConv1D, GlobalAveragePooling1D
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.optimizers import Adam


def MLP(layers, units, dropout_rate, input_shape, learning_rate=1e-3):
    """Creates an instance of a multi-layer perceptron model.

    Parameters
    ----------
    layers: int
        Number of `Dense` layers in the model.
    units: int
        Output dimension of the layers.
    dropout_rate: float
        Percentage of input to drop at Dropout layers.
    input_shape: tuple
        Shape of input to the model.
    learning_rate: float
        Learning rate parameter for the model

    Returns
    -------
    model:
        An compiled MLP keras model instance.
    """

    model = Sequential()
    model.add(Dropout(dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    return model


def sepCNN(blocks, filters, kernel_size, embedding_dim, dropout_rate, pool_size,
           input_shape, num_features, pretrained_embedding=False,
           embedding_trainable=False, embedding_weights=None, learning_rate=1e-3):
    """ Creates an instance of a separable CNN model.

    Parameters
    ----------
    blocks: int
        Number of pairs of sepCNN and pooling blocks in the model. One block
        contains [DropOut, Conv1D, Conv1D, MaxPool]
    filters: int
        Output dimension of the layers.
    kernel_size: int
        Length of the convolution window.
    embedding_dim: int
        Dimension of the embedding vectors.
    dropout_rate: float
        Percentage of input to drop at Dropout layers.
    pool_size: int
        Factor by which to downscale input at MaxPooling layer.
    input_shape: tuple
        Shape of input to the model.
    num_features: int
        Number of words (embedding input dimension).
    pretrained_embedding: bool
        True if pre-trained embedding is on.
    embedding_trainable: bool
        True if embedding layer is trainable.
    embedding_weights: np.ndarray
        Dictionary with embedding coefficients.
    learning_rate: float
        Learning rate parameter for the model

    Returns
    -------
    model:
        A compiled sepCNN keras model instance.
    """

    model = Sequential()

    if pretrained_embedding:
        model.add(Embedding(num_features, embedding_dim,
                            input_length=input_shape[0],
                            embeddings_initializer=Constant(embedding_weights),
                            trainable=embedding_trainable))
    else:
        model.add(Embedding(num_features, embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks-1):
        model.add(Dropout(dropout_rate))
        model.add(SeparableConv1D(filters, kernel_size, activation='relu', padding='same'))
        model.add(SeparableConv1D(filters, kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size))

    model.add(SeparableConv1D(filters*2, kernel_size, activation='relu', padding='same'))
    model.add(SeparableConv1D(filters*2, kernel_size, activation='relu',  padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    return model
