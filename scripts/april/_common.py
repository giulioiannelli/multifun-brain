"""Shared orchestration helpers for the April pipeline scripts.

Single canonical home for: dump results, dump config, dump failed
matrices, build a PipelineConfig from CLI args. The 01/02/03 scripts
are thin and reuse this module to stay readable.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from multifunbrain import PipelineConfig, PipelineResult, run_pipeline
from multifunbrain.datasets.april import AprilEntry, gamma_for, label_for

log = logging.getLogger(__name__)

__all__ = [
    "PipelineRunArgs",
    "add_common_args",
    "config_from_args",
    "dump_failed",
    "dump_results",
    "run_entries",
]


# ──────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────


class PipelineRunArgs(argparse.Namespace):
    out: Path
    gamma: float | None
    sigma: float
    filter_threshold: float | None
    run_rich_club: bool
    run_community_detection: bool


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Common flags for 01/02/03 scripts (output dir + a few config knobs)."""
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (created if needed).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Override the per-entry gamma (p/n). Default: None — gamma is "
            "looked up per processing variant via "
            "multifunbrain.datasets.april.gamma_for()."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="MP noise variance.",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=None,
        help="Override edge threshold (None ⇒ percolation).",
    )
    parser.add_argument(
        "--run-rich-club",
        action="store_true",
        help="Enable rich-club analysis (slow).",
    )
    parser.add_argument(
        "--run-community-detection",
        action="store_true",
        help="Enable Louvain community detection (off by default — LRG partitions are the main output).",
    )


def _make_config(args: PipelineRunArgs, *, gamma: float | None) -> PipelineConfig:
    """Build a PipelineConfig for one entry with a resolved gamma."""
    return PipelineConfig(
        gamma=gamma,
        sigma=args.sigma,
        filter_threshold=args.filter_threshold,
        run_rich_club=args.run_rich_club,
        run_community_detection=args.run_community_detection,
    )


def config_from_args(args: PipelineRunArgs) -> PipelineConfig:
    """Reference PipelineConfig snapshot for ``config.json`` (no per-entry gamma)."""
    return _make_config(args, gamma=args.gamma)


# ──────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────


def dump_results(out_dir: Path, results: Iterable[PipelineResult]) -> None:
    """Pickle a results list + write summary CSV under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    frames = [r.summary_table() for r in results if r.network_analyses]
    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(out_dir / "summary.csv", index=False)


def dump_config(out_dir: Path, config: PipelineConfig) -> None:
    """Write a JSON snapshot of the pipeline config next to results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)


def dump_failed(out_dir: Path, results: Iterable[PipelineResult]) -> None:
    """Write per-matrix errors to ``failed_matrices.json`` (only failures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = [
        {"label": r.label, "error": r.error}
        for r in results
        if r.error is not None
    ]
    with open(out_dir / "failed_matrices.json", "w") as f:
        json.dump(failures, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────


def run_entries(
    entries: list[AprilEntry],
    args: PipelineRunArgs,
) -> int:
    """Run the pipeline on a list of April entries and dump outputs.

    Gamma is resolved **per entry** (different processing variants have
    different ``n_timepoints``). ``--gamma`` on the CLI overrides the
    per-entry lookup for every entry.
    """
    if not entries:
        print("No entries to run.", file=sys.stderr)
        return 1

    print(f"Running pipeline on {len(entries)} matrices → {args.out}")
    results: list[PipelineResult] = []
    for entry in entries:
        gamma = args.gamma if args.gamma is not None else gamma_for(entry)
        config = _make_config(args, gamma=gamma)
        result = run_pipeline(entry.path, config=config, label=label_for(entry))
        results.append(result)

    dump_results(args.out, results)
    dump_config(args.out, config_from_args(args))
    dump_failed(args.out, results)

    n_ok = sum(1 for r in results if r.error is None)
    n_fail = len(results) - n_ok
    print(f"Done: {n_ok}/{len(results)} succeeded, {n_fail} failed.")
    return 0
