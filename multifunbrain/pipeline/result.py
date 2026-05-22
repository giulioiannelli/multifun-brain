"""Pipeline result container + sanitiser helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig

__all__ = ["PipelineResult", "sanitise"]


@dataclass
class PipelineResult:
    """Container for all pipeline outputs.

    Attributes
    ----------
    config : PipelineConfig
        Configuration used for this run.
    label : str or None
        User-provided label for this result.
    n_regions_original : int
        Number of regions in the raw input matrix (before dropping dead regions).
    dropped_regions : np.ndarray
        Indices of dead regions that were dropped (empty array if none).
    corr_raw : np.ndarray
        The input correlation matrix.
    corr_prepared : np.ndarray
        Cleaned (symmetrised, clipped, diagonal zeroed) matrix, with dead
        regions removed.
    descriptive : dict
        Output of :func:`descriptive_report`.
    filtered_networks : dict
        ``{filter_name: {'graph': nx.Graph, 'nodes_removed': list}}``.
    network_analyses : dict
        ``{filter_name: network_summary_report()}``.
    lrg_results : dict
        ``{filter_name: list of partition dicts}``.
    """

    config: PipelineConfig
    label: str | None = None
    n_regions_original: int = 0
    dropped_regions: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    corr_raw: np.ndarray | None = None
    corr_prepared: np.ndarray | None = None
    descriptive: dict[str, Any] | None = None
    filtered_networks: dict[str, dict[str, Any]] = field(default_factory=dict)
    network_analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    lrg_results: dict[str, list[dict]] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise results to a plain dict (DataFrames become dicts)."""
        d: dict[str, Any] = {
            "label": self.label,
            "error": self.error,
            "n_regions_original": self.n_regions_original,
            "n_regions_after_cleanup": int(self.n_regions_original - len(self.dropped_regions)),
            "dropped_regions": self.dropped_regions.tolist(),
            "config": self.config.__dict__,
            "descriptive": sanitise(self.descriptive),
            "filtered_networks": {
                name: {
                    "n_nodes": info["graph"].number_of_nodes(),
                    "n_edges": info["graph"].number_of_edges(),
                    "nodes_removed": info["nodes_removed"],
                    **(
                        {"percolation": {
                            k: v for k, v in info["percolation"].items()
                            if k != "max_per_node"
                        }}
                        if info.get("percolation") else {}
                    ),
                }
                for name, info in self.filtered_networks.items()
            },
            "network_analyses": {
                name: sanitise(analysis)
                for name, analysis in self.network_analyses.items()
            },
            "lrg_results": {
                name: (
                    parts
                    if isinstance(parts, dict) and "error" in parts
                    else [
                        {"tau": p["tau"], "n_clusters": len(np.unique(p["partition"]))}
                        for p in parts
                    ]
                )
                for name, parts in self.lrg_results.items()
            },
        }
        return d

    def summary_table(self) -> pd.DataFrame:
        """One-row DataFrame of key metrics for easy concatenation across runs."""
        rows = []
        for fname, analysis in self.network_analyses.items():
            gm = analysis.get("global_metrics", {})
            comm = analysis.get("community", {})
            row = {
                "label": self.label,
                "n_regions_original": self.n_regions_original,
                "n_dropped": len(self.dropped_regions),
                "filter": fname,
                "n_nodes": gm.get("n_nodes"),
                "n_edges": gm.get("n_edges"),
                "density": gm.get("density"),
                "avg_clustering": gm.get("avg_clustering"),
                "avg_shortest_path": gm.get("avg_shortest_path"),
                "global_efficiency": gm.get("global_efficiency"),
                "assortativity": gm.get("assortativity"),
                "modularity": comm.get("modularity"),
                "n_communities": comm.get("n_communities"),
            }
            rows.append(row)
        return pd.DataFrame(rows)


def sanitise(obj: Any) -> Any:
    """Recursively convert numpy arrays and DataFrames for JSON-like output."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj
