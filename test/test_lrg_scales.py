"""Tests for :mod:`multifunbrain.analysis.lrg.scales`."""

from __future__ import annotations

import numpy as np
import pytest

from multifunbrain.analysis.lrg.scales import characteristic_scales


def test_tau_prime_recovers_inverse_lambda_max():
    eigenvalues = np.array([0.0, 0.5, 1.0, 4.0])
    time_grid = np.logspace(-2, 2, 50)
    specific_heat = np.exp(-((np.log10(time_grid)) ** 2))
    out = characteristic_scales(eigenvalues, time_grid, specific_heat)
    assert out["tau_prime"] == pytest.approx(0.25)


def test_tau_star_at_specific_heat_peak():
    tau = np.logspace(-2, 5, 400)
    c = np.exp(-((np.log10(tau) - 2.0) ** 2))
    eigenvalues = np.array([0.5, 1.0, 2.0])
    out = characteristic_scales(eigenvalues, tau, c)
    assert np.log10(out["tau_star"]) == pytest.approx(2.0, abs=0.05)


def test_default_tau_grid_spans_prime_to_post_star():
    eigenvalues = np.array([0.5, 1.0, 4.0])
    tau = np.logspace(-2, 5, 400)
    c = np.exp(-((np.log10(tau) - 1.0) ** 2))
    out = characteristic_scales(eigenvalues, tau, c, n_tau=20)
    assert out["tau_grid_default"][0] == pytest.approx(out["tau_prime"])
    assert out["tau_grid_default"][-1] >= out["tau_star"] * 2.5
    assert out["tau_grid_default"].size == 20


def test_no_nonzero_eigs_raises():
    with pytest.raises(ValueError):
        characteristic_scales(
            np.zeros(5),
            np.logspace(-2, 5, 50),
            np.zeros(50),
        )
