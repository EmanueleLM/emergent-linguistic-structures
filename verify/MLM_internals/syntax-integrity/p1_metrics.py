import numpy as np

from scipy.stats import spearmanr

def same_distance(M, M_hat, max_length):
    """
    Exact number of times the distance of non-pad words in M and M_hat are equal 
    (up to a round of the first decimal digit).
    """
    eq, tot_length = 0, 0
    for j in range(len(M)):
        m = M[j].reshape(max_length, max_length)
        m_hat = M_hat[j].reshape(max_length, max_length)
        diff = (np.round(m_hat)-m.astype(int)).flatten()
        eq += len(diff[diff==0])
        tot_length += len(diff)
    if tot_length > 0:
        return eq/tot_length
    else:
        return 0.

def spearman_pairwise(M, M_hat, max_length):
    """
    Matrix2matrix spearman
    For this coefficient we consider also the pad-tokens (so it can be refined).
    """
    C = []
    for j in range(len(M)):
        m_hat_flat, m_flat = np.round(M_hat[j].flatten()),  M[j].flatten()
        m_hat_flat[m_hat_flat != 1] = 0
        m_flat[m_flat != 1] = 0
        coeff = spearmanr(m_flat, m_hat_flat)
        if np.isnan(coeff.correlation):
            pass
        else:
            C += [coeff.correlation]
    if len(C) > 0:
        mean, std = np.array(C).mean(), np.array(C).std()
    else:
        mean, std = 0., 0.
    return mean, std

def spearman(M, M_hat, max_length, adjacency=False):
    """
    Vectors version of spearman
    For this coefficient we consider also the pad-tokens (so it can be refined).
    adjacency considers only the adjacency matrices
    """
    M_flat, M_hat_flat = [], []
    for j in range(len(M)):
        m_hat_flat, m_flat = np.round(M_hat[j].flatten()),  M[j].flatten()
        if adjacency is True:
            m_hat_flat[m_hat_flat != 1] = 0
            m_flat[m_flat != 1] = 0
        M_hat_flat += [m for m in m_hat_flat]
        M_flat += [m for m in m_flat]
    coeff = spearmanr(np.array(M_hat_flat).flatten(), np.array(M_flat).flatten())
    return coeff.correlation

def UUAS(M, M_hat, max_length):
    """
    Calculate the UUAS (should be correct now).
    """
    eq, tot_length = 0, 0
    for j in range(len(M)):
        m = M[j].reshape(max_length, max_length)
        m_hat = np.round(M_hat[j])
        m_hat_direct_edges = set(np.argwhere(m_hat.flatten() == 1).flatten().tolist())
        m_direct_edges = set(np.argwhere(m.flatten() == 1).flatten().tolist())
        diff = m_direct_edges.intersection(m_hat_direct_edges)
        eq += len(diff)
        tot_length += len(m_direct_edges)
        #print(m_set, m_hat_set)
    if tot_length > 0:
        return eq/tot_length
    else:
        return 0.
