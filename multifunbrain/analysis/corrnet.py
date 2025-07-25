"""Simple correlation network helpers."""

import numpy as np


def compute_correlation_matrix(timeseries):
    """
    Compute the Pearson correlation matrix from multivariate time series data.

    Parameters:
        timeseries (ndarray): Array of shape (n_regions, n_timepoints)

    Returns:
        corr_matrix (ndarray): Correlation matrix of shape (n_regions, n_regions)
    """
    return np.corrcoef(timeseries)


def marchenko_pastur(l, g):
    """
    Compute the Marchenko-Pastur distribution density.

    Parameters
    ----------
    l : array_like
        Input eigenvalue(s) at which the density is evaluated.
    g : float
        Aspect ratio parameter, typically defined as N/M or M/N.

    Returns
    -------
    ndarray
        The Marchenko-Pastur density evaluated at `l`.

    Notes
    -----
    The density is defined as:

        p(l) = (1 / (2 * π * g * l)) * sqrt( max((1+√g)² - l, 0) * max(l - (1-√g)², 0) )

    Reference
    ---------
    Marchenko, V. A. and Pastur, L. A. (1967). "Distribution of eigenvalues for some sets 
    of random matrices." Mathematics of the USSR-Sbornik, 1(4), 457-483.
    """
    def m0(a):
        """Compute the element-wise maximum of the array `a` and 0."""
        return np.maximum(a, 0)

    g_plus = (1 + np.sqrt(g))**2
    g_minus = (1 - np.sqrt(g))**2

    density = np.sqrt(m0(g_plus - l) * m0(l - g_minus)) / (2 * np.pi * g * l)
    return density
#
# Implement the correct Marchenko-Pastur density function
def marchenko_pastur_density(x, gamma, sigma=1):
    """
    Marchenko-Pastur distribution density function
    x: eigenvalue
    gamma: aspect ratio p/n (samples/features)
    sigma: noise variance (default=1)
    """
    if gamma <= 1:
        lambda_min = sigma * (1 - np.sqrt(gamma))**2
        lambda_max = sigma * (1 + np.sqrt(gamma))**2
    else:
        lambda_min = sigma * (1 - np.sqrt(1/gamma))**2
        lambda_max = sigma * (1 + np.sqrt(1/gamma))**2
    
    # Only defined in the support interval
    mask = (x >= lambda_min) & (x <= lambda_max)
    density = np.zeros_like(x)
    
    x_valid = x[mask]
    if len(x_valid) > 0:
        density[mask] = (1 / (2 * np.pi * sigma * gamma * x_valid)) * \
                       np.sqrt((lambda_max - x_valid) * (x_valid - lambda_min))
    
    return density