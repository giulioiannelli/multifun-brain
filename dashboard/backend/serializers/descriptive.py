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


def precision_spec(result, **_) -> dict:
    """Precision matrix (inverse correlation) from the precision section."""
    prec = (result.descriptive or {}).get("precision") or {}
    mat = prec.get("precision_matrix")
    if mat is None:
        return {"kind": "precision", "label": result.label, "error": "no precision matrix"}
    mat = np.asarray(mat)
    bound = float(np.nanmax(np.abs(mat))) or 1.0
    spec = _matrix_spec(result, mat, "precision", zmin=-bound, zmax=bound)
    spec["method"] = prec.get("method")
    spec["sparsity"] = prec.get("sparsity")
    return clean(spec)


def _kde_curve(samples, lo: float, hi: float, n_used: int, binwidth: float, n_grid: int = 256):
    """Gaussian-KDE density over ``[lo, hi]``, scaled to histogram counts.

    Returns ``(xs, ys)`` where ``ys`` is the density times ``n_used * binwidth``
    so the smooth curve overlays the count histogram directly. ``None, None`` if
    the sample is too small/degenerate to estimate.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    s = s[(s >= lo) & (s <= hi)]
    if s.size < 5 or float(np.ptp(s)) <= 0 or not binwidth or n_used <= 0:
        return None, None
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(s)
    except Exception:  # noqa: BLE001 - degenerate covariance etc.; just skip the overlay
        return None, None
    xs = np.linspace(lo, hi, n_grid)
    ys = kde(xs) * (n_used * binwidth)
    return xs, ys


def weights_spec(result, **_) -> dict:
    """Weight-distribution histogram (signed) + summary statistics + KDE overlay."""
    wd = (result.descriptive or {}).get("weight_distribution") or {}
    hist = wd.get("histogram")
    counts, edges = (hist[0], hist[1]) if hist is not None else (None, None)

    kde_x = kde_y = None
    if edges is not None and counts is not None and len(counts):
        e = np.asarray(edges, dtype=float)
        lo, hi = float(e[0]), float(e[-1])
        nb = len(counts)
        binwidth = (hi - lo) / nb if nb else 0.0
        kde_x, kde_y = _kde_curve(
            wd.get("all_weights"), lo, hi, int(np.sum(counts)), binwidth
        )

    return clean(
        {
            "kind": "weights",
            "label": result.label,
            "counts": counts,
            "edges": edges,
            "kde_x": kde_x,
            "kde_y": kde_y,
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


def _adaptive_spectrum_hist(eigs) -> dict:
    """Density-adaptive histogram of an eigenvalue spectrum.

    Bin width follows the bulk density via the Freedman-Diaconis rule
    (∝ IQR·n^-1/3). When a heavy tail (a few isolated signal eigenvalues) would
    blow the bin count up and flatten the bulk into one bar, the bulk is binned
    over a robust window (≤ 99th percentile) and the extreme eigenvalues are
    reported separately (``n_above``/``above``) instead of being binned in.
    """
    e = np.asarray(eigs, dtype=float)
    e = e[np.isfinite(e)]
    empty = {"counts": None, "edges": None, "n_above": 0, "above": []}
    if e.size < 2:
        return empty

    q25, q75 = np.percentile(e, [25, 75])
    iqr = float(q75 - q25)
    n = int(e.size)
    width = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else 0.0
    span = float(e.max() - e.min())
    nbins_full = int(np.ceil(span / width)) if width > 0 else 0

    # Well-behaved spectrum: plain Freedman-Diaconis over the full range.
    if 0 < nbins_full <= 200:
        counts, edges = np.histogram(e, bins="fd")
        return {"counts": counts, "edges": edges, "n_above": 0, "above": []}

    # Heavy tail: focus on the bulk, surface the extreme eigenvalues as markers.
    hi = float(np.percentile(e, 99))
    if hi <= float(e.min()):
        hi = float(np.median(e))
    bulk = e[e <= hi]
    if bulk.size >= 2 and iqr > 0:
        edges = np.histogram_bin_edges(bulk, bins="fd")
        if edges.size > 51:  # cap at ~50 bins so a degenerate spike + sparse tail stays readable
            edges = np.linspace(float(bulk.min()), hi, 51)
    else:
        edges = np.linspace(float(e.min()), hi, 41)
    counts, edges = np.histogram(bulk, bins=edges)
    above = np.sort(e[e > hi])[::-1]
    return {
        "counts": counts,
        "edges": edges,
        "n_above": int(above.size),
        "above": [float(x) for x in above[:8]],
    }


def spectrum_spec(result, **_) -> dict:
    """Correlation eigenvalue spectrum with Marchenko-Pastur bulk bounds.

    Ships a precomputed density-adaptive histogram (``counts``/``edges``) so the
    bulk is resolved even when one large signal eigenvalue dominates the range.
    """
    sp = (result.descriptive or {}).get("spectrum") or {}
    hist = _adaptive_spectrum_hist(sp.get("eigenvalues")) if sp.get("eigenvalues") is not None else {
        "counts": None, "edges": None, "n_above": 0, "above": [],
    }

    kde_x = kde_y = None
    if hist["edges"] is not None and hist["counts"] is not None and len(hist["counts"]):
        e = np.asarray(hist["edges"], dtype=float)
        lo, hi = float(e[0]), float(e[-1])
        nb = len(hist["counts"])
        binwidth = (hi - lo) / nb if nb else 0.0
        kde_x, kde_y = _kde_curve(
            sp.get("eigenvalues"), lo, hi, int(np.sum(hist["counts"])), binwidth
        )

    return clean(
        {
            "kind": "spectrum",
            "label": result.label,
            "eigenvalues": sp.get("eigenvalues"),
            "counts": hist["counts"],
            "edges": hist["edges"],
            "kde_x": kde_x,
            "kde_y": kde_y,
            "n_above": hist["n_above"],
            "above": hist["above"],
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
