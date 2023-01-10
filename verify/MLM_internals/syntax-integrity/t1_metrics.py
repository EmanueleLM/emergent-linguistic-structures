import numpy as np

def accuracy(Y, Y_hat, classification=False):
    """
    For classification the input is assumed to be (len, max_length, num_classes)
    """
    if classification is False:
        Y, Y_hat = Y.flatten(), np.round(Y_hat.flatten())
    else:  # assume one-hot and 3D-shape (len, max_length, num_classes)
        if Y.ndim == 2:
            Y = Y.reshape(1, Y.shape[0], Y.shape[1])
        if Y_hat.ndim == 2:
            Y_hat = Y_hat.reshape(1, Y_hat.shape[0], Y_hat.shape[1])
        Y = Y.reshape(Y.shape[0]*Y.shape[1], -1)       
        Y_hat = Y_hat.reshape(Y_hat.shape[0]*Y_hat.shape[1], -1)
        Y = np.argmax(Y, axis=1)
        Y_hat = np.argmax(Y_hat, axis=1)
        #print(Y)
        #print(Y_hat)
        # print(np.sum(np.equal(Y[:100], Y_hat[:100])/100))
    eq = np.sum(np.equal(Y, Y_hat))
    cnt = len(Y)
    return eq/cnt
