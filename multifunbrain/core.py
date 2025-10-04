"""Core helper functions for :mod:`multifunbrain`."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt

__all__ = ["hello_brain", "band_filter", "marchenko_pastur_density"]


def hello_brain(name: str) -> str:
    """Return a friendly greeting.

    Parameters
    ----------
    name:
        Name or identifier that should appear in the greeting.
    """

    return f"Hello, {name}! Welcome to multifun-brain."


def band_filter(
    data: ArrayLike,
    low: float,
    high: float,
    fs: float = 1.0,
    order: int = 4,
    btype: str = "bandpass",
) -> np.ndarray:
    """Apply a Butterworth filter along the last axis of ``data``.

    Parameters
    ----------
    data:
        Array-like signal with the last axis representing time.
    low:
        Lower cutoff frequency in Hz.
    high:
        Upper cutoff frequency in Hz.
    fs:
        Sampling frequency of the data in Hz.
    order:
        Filter order passed to :func:`scipy.signal.butter`.
    btype:
        Filter type, defaults to ``"bandpass"``.

    Returns
    -------
    numpy.ndarray
        Filtered array with the same shape as the input.

    Raises
    ------
    ValueError
        If frequency parameters are inconsistent or outside the Nyquist range.
    """

    if fs <= 0:
        raise ValueError("Sampling frequency 'fs' must be positive.")

    nyq = 0.5 * fs
    if nyq <= 0:
        raise ValueError("Nyquist frequency must be positive.")

    if btype == "bandpass":
        if not 0 < low < high < nyq:
            raise ValueError("For a bandpass filter require 0 < low < high < Nyquist.")
        wn: Iterable[float] = (low / nyq, high / nyq)
    elif btype == "lowpass":
        if not 0 < high < nyq:
            raise ValueError("For a lowpass filter require 0 < high < Nyquist.")
        wn = (high / nyq,)
    elif btype == "highpass":
        if not 0 < low < nyq:
            raise ValueError("For a highpass filter require 0 < low < Nyquist.")
        wn = (low / nyq,)
    else:
        raise ValueError(f"Unsupported filter type: {btype}")

    b, a = butter(order, wn, btype=btype)
    data_arr = np.asarray(data)
    return filtfilt(b, a, data_arr, axis=-1)


def marchenko_pastur_density(
    eigenvalues: ArrayLike,
    gamma: float,
    sigma: float = 1.0,
) -> np.ndarray:
    r"""Evaluate the Marchenko–Pastur density for given eigenvalues.

    Parameters
    ----------
    eigenvalues:
        Iterable of eigenvalues at which to evaluate the density.
    gamma:
        Aspect ratio of the random matrix (``p / n``). Must be positive.
    sigma:
        Noise variance term. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Density values for each eigenvalue. Values outside the support are zero.

    Notes
    -----
    The Marchenko–Pastur distribution has support :math:`[(\\lambda_-), (\\lambda_+)]`
    where :math:`\\lambda_{\\pm} = \sigma (1 \pm \sqrt{\min(1, \gamma)})^2`.
    """

    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    lam = np.asarray(eigenvalues, dtype=float)

    if gamma <= 1:
        lambda_min = sigma * (1 - np.sqrt(gamma)) ** 2
        lambda_max = sigma * (1 + np.sqrt(gamma)) ** 2
    else:
        inv_gamma = 1 / gamma
        lambda_min = sigma * (1 - np.sqrt(inv_gamma)) ** 2
        lambda_max = sigma * (1 + np.sqrt(inv_gamma)) ** 2

    density = np.zeros_like(lam, dtype=float)
    mask = (lam >= lambda_min) & (lam <= lambda_max)
    lam_valid = lam[mask]
    if lam_valid.size:
        density[mask] = (
            1.0
            / (2 * np.pi * sigma * gamma * lam_valid)
            * np.sqrt((lambda_max - lam_valid) * (lam_valid - lambda_min))
        )
    return density
