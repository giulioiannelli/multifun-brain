"""Metastable-node detection across LRG diffusion scales.

A node is *metastable* if the set of other nodes sharing its cluster changes
across consecutive ``tau`` steps. Unlike a raw label-change test, this
formulation is invariant to scipy's arbitrary relabelling between successive
``fcluster`` calls, so it actually tracks structural reassignment.

Per Villegas et al., *Phys. Rev. Research* 7, 013065 (2025), metastable
nodes act as communication bridges between mesoscopic modules and are
identified from the cluster composition of the *optimal* partition at each
``tau``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["metastable_nodes"]


def metastable_nodes(per_tau_results: list[dict]) -> dict[int, dict]:
    """Identify nodes whose cluster-membership *set* changes across ``tau``.

    Parameters
    ----------
    per_tau_results : list of dict
        Each entry has ``'tau'`` and ``'partition'`` (1D label array of
        length ``N``). Should correspond to the *optimal* (``argmax Ψ``)
        partition at each ``tau``.

    Returns
    -------
    dict[int, dict]
        ``{node_index: {'n_switches', 'first_switch_tau', 'last_switch_tau',
        'switch_taus', 'history'}}``, only for nodes that experience at least
        one structural switch.
    """
    if not per_tau_results:
        return {}

    tau_values = [float(r["tau"]) for r in per_tau_results]
    partitions = [np.asarray(r["partition"]) for r in per_tau_results]
    n_nodes = len(partitions[0])

    membership_per_tau = [
        [frozenset(int(j) for j in np.where(part == part[x])[0]) for x in range(n_nodes)]
        for part in partitions
    ]

    out: dict[int, dict] = {}
    for x in range(n_nodes):
        switch_taus = []
        for k in range(1, len(partitions)):
            if membership_per_tau[k][x] != membership_per_tau[k - 1][x]:
                switch_taus.append(tau_values[k])
        if switch_taus:
            history = [(tau_values[k], int(partitions[k][x])) for k in range(len(partitions))]
            out[x] = {
                "n_switches": len(switch_taus),
                "first_switch_tau": switch_taus[0],
                "last_switch_tau": switch_taus[-1],
                "switch_taus": switch_taus,
                "history": history,
            }
    return out
