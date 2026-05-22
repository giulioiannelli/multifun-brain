"""Shared LRG-batch orchestration for the April scripts (04, 05, 06).

Mirrors ``_common.py`` but drives ``run_multiscale_lrg`` instead of the
section-1-3 pipeline. One entry → one ``MultiscaleResult`` per requested
backbone. Outputs land under ``<out_dir>/<backbone>/<label>.pkl`` with a
single ``manifest.csv`` and ``failed_matrices.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from multifunbrain.datasets.april import AprilEntry, gamma_for, label_for
from multifunbrain.pipeline.multiscale import MultiscaleResult, run_multiscale_lrg

log = logging.getLogger(__name__)

__all__ = [
    "LRGRunArgs",
    "add_lrg_args",
    "run_lrg_entries",
]


class LRGRunArgs(argparse.Namespace):
    out: Path
    backbones: list[str]
    alpha: float
    apply_mp_denoise: bool
    no_mp_denoise: bool
    sigma: float
    n_tau: int
    backbone_weights: str
    log10_t1: float
    log10_t2: float
    time_grid_steps: int


def add_lrg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory (created if needed).")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["mp_validated", "percolation"],
        choices=["disparity", "lans", "mp_validated", "percolation"],
        help="Backbone(s) to run per matrix. Default: mp_validated + percolation.",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Backbone significance level (disparity / LANS).")
    parser.add_argument("--no-mp-denoise", action="store_true",
                        help="Skip MP shrinkage before the backbone step.")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-tau", type=int, default=30,
                        help="Number of log-spaced τ values for the per-τ pass.")
    parser.add_argument(
        "--backbone-weights",
        choices=["abs", "positive"],
        default="positive",
        help="Pre-backbone signed-to-nonneg map. Default: positive.",
    )
    parser.add_argument("--log10-t1", type=float, default=-2.0,
                        help="log10(τ_min) for the broad C(τ) grid.")
    parser.add_argument("--log10-t2", type=float, default=5.0,
                        help="log10(τ_max) for the broad C(τ) grid.")
    parser.add_argument("--time-grid-steps", type=int, default=600,
                        help="Steps on the broad C(τ) grid.")


def _label_to_relpath(label: str) -> Path:
    """Turn ``'global/co2_bpfBOLD'`` into ``Path('global/co2_bpfBOLD.pkl')``."""
    return Path(label + ".pkl")


def _save_one(out_dir: Path, backbone: str, result: MultiscaleResult) -> Path:
    target = out_dir / backbone / _label_to_relpath(result.label)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(result, f)
    return target


def _config_snapshot(args: LRGRunArgs) -> dict:
    return {
        "backbones": list(args.backbones),
        "alpha": args.alpha,
        "apply_mp_denoise": not args.no_mp_denoise,
        "sigma": args.sigma,
        "n_tau": args.n_tau,
        "backbone_weights": args.backbone_weights,
        "log10_t1": args.log10_t1,
        "log10_t2": args.log10_t2,
        "time_grid_steps": args.time_grid_steps,
    }


def run_lrg_entries(
    entries: list[AprilEntry],
    args: LRGRunArgs,
) -> int:
    """Run multiscale LRG on a list of entries × backbones; save artefacts."""
    if not entries:
        print("No entries to run.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "config.json", "w") as f:
        json.dump(_config_snapshot(args), f, indent=2)

    manifest_rows: list[dict] = []
    failures: list[dict] = []

    n_total = len(entries) * len(args.backbones)
    print(f"Running multiscale LRG on {len(entries)} matrices × "
          f"{len(args.backbones)} backbones = {n_total} runs → {args.out}")
    t_start = time.time()

    for entry in entries:
        from multifunbrain.datasets.april import load_entry
        gamma = gamma_for(entry)
        label = label_for(entry)
        try:
            corr_raw = load_entry(entry)
        except Exception as exc:
            failures.append({"label": label, "backbone": "<load>",
                             "error": f"load_entry failed: {exc!r}"})
            continue

        for backbone in args.backbones:
            t0 = time.time()
            try:
                result = run_multiscale_lrg(
                    corr_raw,
                    gamma=gamma,
                    backbone=backbone,
                    alpha=args.alpha,
                    backbone_weights=args.backbone_weights,
                    apply_mp_denoise=not args.no_mp_denoise,
                    sigma=args.sigma,
                    n_tau=args.n_tau,
                    time_grid_steps=args.time_grid_steps,
                    log10_t1=args.log10_t1,
                    log10_t2=args.log10_t2,
                    label=label,
                )
            except Exception as exc:  # pragma: no cover - defensive
                failures.append({"label": label, "backbone": backbone,
                                 "error": repr(exc)})
                log.exception("[%s|%s] crashed", label, backbone)
                continue
            dt = time.time() - t0

            if result.error is None:
                path = _save_one(args.out, backbone, result)
                manifest_rows.append({
                    "label": label,
                    "backbone": backbone,
                    "level": entry.level,
                    "contrast": entry.contrast,
                    "processing": entry.processing,
                    "band": entry.band,
                    "subject": entry.subject,
                    "gamma": gamma,
                    "n_alive": len(result.nodes_alive),
                    "tau_prime": result.tau_prime,
                    "tau_star": result.tau_star,
                    "n_metastable": len(result.metastable),
                    "runtime_s": dt,
                    "path": str(path.relative_to(args.out)),
                })
            else:
                failures.append({"label": label, "backbone": backbone,
                                 "error": result.error})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.out / "manifest.csv", index=False)
    with open(args.out / "failed_matrices.json", "w") as f:
        json.dump(failures, f, indent=2)

    elapsed = time.time() - t_start
    n_ok = len(manifest_rows)
    print(f"Done: {n_ok}/{n_total} succeeded, {len(failures)} failed in "
          f"{elapsed:.1f}s. Manifest at {args.out / 'manifest.csv'}.")
    return 0 if n_ok > 0 else 1
