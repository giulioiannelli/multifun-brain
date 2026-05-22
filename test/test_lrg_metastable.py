"""Tests for :mod:`multifunbrain.analysis.lrg.metastable`."""

from __future__ import annotations

import numpy as np

from multifunbrain.analysis.lrg.metastable import metastable_nodes


def _entry(tau, partition):
    return {"tau": tau, "partition": np.asarray(partition)}


def test_no_switches_returns_empty():
    p = [0, 0, 1, 1, 2, 2]
    per_tau = [_entry(t, p) for t in [0.1, 1.0, 10.0]]
    assert metastable_nodes(per_tau) == {}


def test_single_bridge_node_is_flagged():
    # Node 2 starts with the {0,1,2} group, then leaves to merge into {3,4,5,2}
    p0 = [0, 0, 0, 1, 1, 1]
    p1 = [0, 0, 1, 1, 1, 1]
    per_tau = [_entry(0.5, p0), _entry(2.0, p1)]
    out = metastable_nodes(per_tau)
    assert 2 in out
    assert out[2]["first_switch_tau"] == 2.0
    assert out[2]["n_switches"] == 1


def test_partition_label_permutation_alone_does_not_flag():
    # Same membership sets, different bare labels — must NOT flag anyone.
    p0 = [0, 0, 0, 1, 1, 1]
    p1 = [9, 9, 9, 7, 7, 7]
    per_tau = [_entry(0.5, p0), _entry(2.0, p1)]
    assert metastable_nodes(per_tau) == {}


def test_empty_input():
    assert metastable_nodes([]) == {}
