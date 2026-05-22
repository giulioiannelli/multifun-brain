"""Generators for creating synthetic graphs and time series."""

from .generators import (
    generate_brain_timeseries,
    generate_brain_timeseries_2,
    generate_brain_timeseries_3,
    generate_brain_timeseries_4,
    generate_brain_timeseries_5,
    generate_flower_graph,
    generate_hierarchical_modular_network,
    generate_hierarchical_modular_pa,
    generate_hierarchical_multiscale_graph,
    generate_hmn,
    generate_multiscale_graph,
)

__all__ = [
    "generate_hmn",
    "generate_flower_graph",
    "generate_brain_timeseries",
    "generate_brain_timeseries_2",
    "generate_brain_timeseries_3",
    "generate_brain_timeseries_4",
    "generate_brain_timeseries_5",
    "generate_multiscale_graph",
    "generate_hierarchical_multiscale_graph",
    "generate_hierarchical_modular_pa",
    "generate_hierarchical_modular_network",
]
