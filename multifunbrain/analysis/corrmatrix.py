"""Back-compat shim — symbols redistributed to scoped subpackages.

This module used to combine I/O, preprocessing, LRG clustering, and
partition comparison. Each function now has a single canonical home:

==========================================  ===================================
Symbol                                       Canonical module
==========================================  ===================================
``load_correlation_matrix``                  :mod:`multifunbrain.io.corrmatrix`
``detect_dead_regions``                      :mod:`multifunbrain.preprocessing.dead_regions`
``prepare_correlation_matrix``               :mod:`multifunbrain.preprocessing.prepare`
``marchenko_pastur_denoise``                 :mod:`multifunbrain.preprocessing.denoising`
``hierarchical_partitions_from_corr``        :mod:`multifunbrain.analysis.lrg.partitions`
``adjusted_rand_index``                      :mod:`multifunbrain.analysis.partition`
``compare_partition_sets``                   :mod:`multifunbrain.analysis.partition`
==========================================  ===================================

This module re-exports them so legacy imports
(``from multifunbrain.analysis.corrmatrix import ...``) keep working.
"""

from __future__ import annotations

from ..io.corrmatrix import load_correlation_matrix
from ..preprocessing.dead_regions import detect_dead_regions
from ..preprocessing.denoising import marchenko_pastur_denoise
from ..preprocessing.prepare import prepare_correlation_matrix
from .lrg.partitions import hierarchical_partitions_from_corr
from .partition import adjusted_rand_index, compare_partition_sets

__all__ = [
    "adjusted_rand_index",
    "compare_partition_sets",
    "detect_dead_regions",
    "hierarchical_partitions_from_corr",
    "load_correlation_matrix",
    "marchenko_pastur_denoise",
    "prepare_correlation_matrix",
]
