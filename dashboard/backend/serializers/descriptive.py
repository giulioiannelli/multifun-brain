"""Serializers for Section-1 (descriptive) plots.

Each function takes a :class:`~multifunbrain.pipeline.result.PipelineResult` and
returns a JSON-safe dict the frontend renders interactively. Node identity always
flows through :mod:`dashboard.backend.remap` so atlas hover labels stay aligned
after dead-region dropping. Serializers accept ``**_`` so the route can pass
shared params (filter, tau_index, ...) uniformly.
"""

from __future__ import annotations

import numpy as np

from ..encode import clean
from ..remap import surviving_labels


def _matrix_spec(result, matrix, kind: str, *, zmin: float, zmax: float) -> dict:
    """Shared heatmap payload: matrix values + atlas node descriptors."""
    matrix = np.asarray(matrix)
    n = int(matrix.shape[0])
    nodes = surviving_labels(result)
    if len(nodes) != n:
        nodes = [
            {"index": i, "name": f"node-{i}", "short": f"node-{i}",
             "hemisphere": "?", "network": "?", "color": "#888888"}
            for i in range(n)
        ]
    return clean(
        {
            "kind": kind,
            "label": result.label,
            "n": n,
            "z": matrix,
            "nodes": nodes,
            "names": [nd["short"] for nd in nodes],
            "networks": [nd["network"] for nd in nodes],
            "colors": [nd["color"] for nd in nodes],
            "zmin": zmin,
            "zmax": zmax,
        }
    )


def heatmap_spec(result, **_) -> dict:
    """Prepared (signed) correlation matrix."""
    if result.corr_prepared is None:
        return {"kind": "heatmap", "label": result.label, "error": "no corr_prepared stored"}
    return _matrix_spec(result, result.corr_prepared, "heatmap", zmin=-1.0, zmax=1.0)


def partial_correlation_spec(result, **_) -> dict:
    """Partial-correlation matrix from the precision section."""
    prec = (result.descriptive or {}).get("precision") or {}
    pc = prec.get("partial_correlations")
    if pc is None:
        return {"kind": "partial_correlation", "label": result.label, "error": "no partial correlations"}
    pc = np.asarray(pc)
    bound = float(np.nanmax(np.abs(pc))) or 1.0
    spec = _matrix_spec(result, pc, "partial_correlation", zmin=-bound, zmax=bound)
    spec["method"] = prec.get("method")
    spec["sparsity"] = prec.get("sparsity")
    return clean(spec)


def weights_spec(result, **_) -> dict:
    """Weight-distribution histogram (signed) + summary statistics."""
    wd = (result.descriptive or {}).get("weight_distribution") or {}
    hist = wd.get("histogram")
    counts, edges = (hist[0], hist[1]) if hist is not None else (None, None)
    return clean(
        {
            "kind": "weights",
            "label": result.label,
            "counts": counts,
            "edges": edges,
            "n_positive": wd.get("n_positive"),
            "n_negative": wd.get("n_negative"),
            "frac_positive": wd.get("frac_positive"),
            "frac_negative": wd.get("frac_negative"),
            "mean": wd.get("mean"),
            "std": wd.get("std"),
            "skewness": wd.get("skewness"),
            "kurtosis": wd.get("kurtosis"),
            "median": wd.get("median"),
        }
    )


def spectrum_spec(result, **_) -> dict:
    """Correlation eigenvalue spectrum with Marchenko-Pastur bulk bounds."""
    sp = (result.descriptive or {}).get("spectrum") or {}
    return clean(
        {
            "kind": "spectrum",
            "label": result.label,
            "eigenvalues": sp.get("eigenvalues"),
            "largest_eigenvalue": sp.get("largest_eigenvalue"),
            "mp_lambda_plus": sp.get("mp_lambda_plus"),
            "mp_lambda_minus": sp.get("mp_lambda_minus"),
            "n_signal": sp.get("n_signal"),
            "n_noise": sp.get("n_noise"),
            "explained_variance_ratio": sp.get("explained_variance_ratio"),
        }
    )


def signed_laplacian_spec(result, **_) -> dict:
    """Signed-Laplacian eigenvalues (coloured by sign) + frustration metrics."""
    sl = (result.descriptive or {}).get("signed_laplacian") or {}
    return clean(
        {
            "kind": "signed_laplacian",
            "label": result.label,
            "eigenvalues": sl.get("eigenvalues"),
            "n_negative_eigenvalues": sl.get("n_negative_eigenvalues"),
            "spectral_gap": sl.get("spectral_gap"),
            "frustration_index": sl.get("frustration_index"),
        }
    )


def signed_balance_spec(result, **_) -> dict:
    """Per-node positive vs negative strength + global balance ratio."""
    nm = (result.descriptive or {}).get("network_metrics") or {}
    nodes = surviving_labels(result)
    return clean(
        {
            "kind": "signed_balance",
            "label": result.label,
            "names": [nd["short"] for nd in nodes],
            "networks": [nd["network"] for nd in nodes],
            "colors": [nd["color"] for nd in nodes],
            "strength_positive": nm.get("strength_positive"),
            "strength_negative": nm.get("strength_negative"),
            "balance_ratio": nm.get("balance_ratio"),
            "total_positive_weight": nm.get("total_positive_weight"),
            "total_negative_weight": nm.get("total_negative_weight"),
        }
    )
