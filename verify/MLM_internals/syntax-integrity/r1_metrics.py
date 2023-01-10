import numpy as np

def accuracy(Y, Y_hat):
    """
    For classification the predicted labels Y_hat is assumed to be a (len, max_length) vector, i.e., one-hot encoded.
    The target label is of the same shape (len, max_length).
    """
    # Single 1-D batch
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    if Y_hat.ndim == 1:
        Y_hat = Y_hat.reshape(1, -1)
    # assume one-hot and 2D-shape (len, max_length)
    a_r1_pred = np.argmax(Y_hat, axis=1)
    a_r1_labels = np.argmax(Y, axis=1)
    eq = np.sum(np.equal(a_r1_pred, a_r1_labels))
    cnt = len(a_r1_pred) 
    return eq/cnt
