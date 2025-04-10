from .shared import *

def hello_brain(name):
    return f"Hello, {name}! Welcome to multifun-brain."

def bandpass_filter(data, low, high, fs=1, order=4, btype='band'):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype=btype)
    return filtfilt(b, a, data)

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