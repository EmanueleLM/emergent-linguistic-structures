import numpy as np

from scipy.stats import spearmanr

def same_distance(Y, Y_hat):
    """
    Y is a one hot-encoded matrix in the form (num_samples, num-classes), Y is an array of numbers, whose shape is (num_samples,)
    The exact distance ratio is the number of time the prediction (via column-wise argmax) is equal to the label (int)
    """ 
    assert Y.shape == Y_hat.shape
    eq = np.sum(np.equal(np.argmax(Y, axis=1).flatten(), np.argmax(Y_hat, axis=1).flatten().astype(int)))
    cnt = len(Y)
    return eq/cnt

def spearman(Y, Y_hat):
    """
    Flatten vectors Spearman correlation
    Y and Y_hat are arrays of numbers, whose shape is (num_samples,)
    """
    assert Y.shape == Y_hat.shape
    coeff = spearmanr(np.argmax(Y, axis=1).flatten(), np.argmax(Y_hat, axis=1).flatten().astype(int))
    return coeff.correlation