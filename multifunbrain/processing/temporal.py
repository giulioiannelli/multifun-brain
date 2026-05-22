"""Temporal signal filtering for fMRI time series.

Wraps :func:`scipy.signal.butter` + :func:`filtfilt` with Butterworth
band/low/highpass filters along the last axis of an input array.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt

__all__ = ["band_filter"]


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
    data : array-like
        Signal with the last axis representing time.
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency of the data in Hz.
    order : int
        Filter order passed to :func:`scipy.signal.butter`.
    btype : str
        Filter type. One of ``"bandpass"`` (default), ``"lowpass"``, ``"highpass"``.

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
