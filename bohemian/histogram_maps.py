import numpy as np
from scipy.stats import rankdata


def inverse_density(X):
    """

    :param X:
    :return:
    """
    unique, counts = np.unique(X, return_counts=True)
    cumcount = np.cumsum(counts)
    cumcount = (cumcount - cumcount.min()) / (cumcount.max() - cumcount.min())
    Z = np.vectorize(dict(zip(unique, cumcount)).get)(X)
    Z[X <= 0] = -np.inf
    return Z


# Rank-normalize histogram?
def density_rank(X):
    """

    :param X:
    :return:
    """
    # X is an array

    # Rank the values in X
    R = rankdata(X[X > 0], method='dense')
    R = (R - 1) / (R.max() - 1)

    # Output array
    Z = np.zeros_like(X)
    Z[X > 0] = R
    Z[X <= 0] = -np.inf

    return Z
