"""LRG diffusion kernel: Laplacian, density matrix, spectral entropy.

The Laplacian Renormalisation Group (LRG) treats the graph as a closed
quantum system in imaginary time. The diffusion kernel
``rho(tau) = exp(-tau * L) / Tr(exp(-tau * L))`` plays the role of a
density matrix; its von-Neumann entropy traces the information loss
under diffusion.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.linalg import expm

__all__ = ["graph_laplacian_and_spectrum", "rho_matrix", "entropy"]


def graph_laplacian_and_spectrum(graph, weight="weight", normalized=False):
    """Assemble the graph Laplacian and its spectrum.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    weight : str or None, optional
        Edge attribute to use as weight. Pass ``None`` for unweighted graphs.
    normalized : bool, optional
        If ``True`` use the symmetric normalised Laplacian; otherwise the
        combinatorial Laplacian.

    Returns
    -------
    L : np.ndarray
        Dense Laplacian matrix with nodes ordered as ``graph.nodes()``.
    spectrum : np.ndarray
        Eigenvalues of ``L`` sorted in ascending order.
    """
    if graph.number_of_nodes() == 0:
        return np.zeros((0, 0), dtype=float), np.array([], dtype=float)

    if normalized:
        laplacian = nx.normalized_laplacian_matrix(graph, weight=weight)
    else:
        laplacian = nx.laplacian_matrix(graph, weight=weight)

    L = laplacian.astype(float).toarray()
    spectrum = np.linalg.eigvalsh(L)
    return L, spectrum


def rho_matrix(tau, L):
    """Normalized diffusion kernel ``rho(tau) = exp(-tau * L) / Tr(...)``.

    Parameters
    ----------
    tau : float
        Diffusion time scale.
    L : ndarray (n x n)
        Graph Laplacian matrix (assumed symmetric and positive semi-definite).

    Returns
    -------
    rho : ndarray (n x n)
        Normalised diffusion matrix with trace 1.
    """
    kernel = expm(-tau * L)
    trace = np.trace(kernel)
    return kernel / trace


def entropy(w, steps=600, t1=-2, t2=5):
    """Diffusion-based spectral entropy S(t) and specific heat C(t).

    Parameters
    ----------
    w : ndarray
        Eigenvalue spectrum of the Laplacian.
    steps : int
        Number of diffusion time steps.
    t1, t2 : float
        ``log10(start time)``, ``log10(end time)``.

    Returns
    -------
    one_minus_S : ndarray
        ``1 - S(t)`` (information remaining after diffusion).
    dS : ndarray
        Specific heat ``C(t) = d(1-S)/d log(t)``.
    VarL : ndarray
        Laplacian spectral variance over time.
    t : ndarray
        Time points (logspaced).
    """
    N = len(w)
    t = np.logspace(t1, t2, int(steps))
    S = np.zeros(len(t))
    VarL = np.zeros(len(t))

    for i, tau in enumerate(t):
        rhoTr = np.exp(-tau * w)
        Tr = np.sum(rhoTr)
        rho = rhoTr / Tr

        S[i] = -np.nansum(rho * np.log(rho)) / np.log(N)

        avg = np.sum(w * rhoTr) / Tr
        avg2 = np.sum((w ** 2) * rhoTr) / Tr
        VarL[i] = avg2 - avg**2

    dS = np.log(N) * np.diff(1 - S) / np.diff(np.log(t))
    return 1 - S, dS, VarL, t
