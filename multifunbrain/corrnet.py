from .core import *


def compute_correlation_matrix(timeseries):
    """
    Compute the Pearson correlation matrix from multivariate time series data.

    Parameters:
        timeseries (ndarray): Array of shape (n_regions, n_timepoints)

    Returns:
        corr_matrix (ndarray): Correlation matrix of shape (n_regions, n_regions)
    """
    return np.corrcoef(timeseries)

