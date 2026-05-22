"""CLI thin-wrapper around the figure builders in ``_handoff_figs``.

Two figures per correlation matrix:

- **fig 1** (``<label>_fig1_network_lrg.pdf``): 4 × 3 grid with the
  network descriptive panels plus a bottom row for the LANS+LRG
  hierarchy summary.
- **fig 2** (``<label>_fig2_dendrogram_cuts.pdf``): 5 × 3 grid with
  one row per k ∈ {2, 5, 7, 17, 31}.

Same defaults as the previous version of this script — ``--single`` for
ad-hoc runs, no flag = full batch over ``freq-contrast-global`` and
``freq-contrast-inter``. The handoff driver ``09_handoff_batch.py`` calls
the builders directly instead of going through this CLI.
"""

from __future__ import annotations

import pickle
import sys
import time
import traceback
from pathlib import Path

from _fig_common import RESULTS_BASE, SRC_BASE, gamma_for_variant
from _handoff_figs import build_fig1, build_fig2

OUT_BASE = RESULTS_BASE / "_example_figs"


def _run_single(
    pkl_path: Path,
    out_dir: Path,
    only: set[str] | None = None,
    force: bool = False,
) -> None:
    label = pkl_path.stem
    gamma = gamma_for_variant(pkl_path.parent.name)
    with open(pkl_path, "rb") as f:
        corr_raw = pickle.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, fn, suffix in [
        ("fig1", build_fig1, "fig1_network_lrg"),
        ("fig2", build_fig2, "fig2_dendrogram_cuts"),
    ]:
        if only and key not in only:
            continue
        out_path = out_dir / f"{label}_{suffix}.pdf"
        if out_path.exists() and not force:
            print(f"    skip (exists): {out_path.name}")
            continue
        t0 = time.time()
        try:
            fn(corr_raw, label, gamma, out_path)
            print(f"    {out_path.name}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(
                f"    [ERR] {key}: {type(e).__name__}: {e}", file=sys.stderr
            )
            traceback.print_exc(file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fig1 (network + LRG descriptive) and fig2 "
        "(dendrogram cuts at fixed k) for April-batch matrices.",
    )
    parser.add_argument(
        "--single", type=Path, default=None,
        help="Run on a single .pkl. Outputs to <out> if given, else OUT_BASE.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--only", nargs="+", choices=["fig1", "fig2"], default=None,
        help="Only render the listed figures (default: both).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-render even if the target PDF already exists.",
    )
    args = parser.parse_args(argv)

    only = set(args.only) if args.only else None

    if args.single is not None:
        out_dir = args.out if args.out is not None else OUT_BASE
        print(f"=== single: {args.single} ===")
        _run_single(
            args.single.resolve(), out_dir.resolve(),
            only=only, force=args.force,
        )
        return 0

    targets = [
        (SRC_BASE / "freq-contrast-global", OUT_BASE / "freq-contrast-global"),
        (SRC_BASE / "freq-contrast-inter", OUT_BASE / "freq-contrast-inter"),
    ]
    overall_t0 = time.time()
    for src_root, out_root in targets:
        if args.out is not None:
            out_root = args.out / src_root.name
        print(f"\n=== {src_root.name} ===")
        pkls = sorted(src_root.rglob("*.pkl"))
        for i, pkl in enumerate(pkls, 1):
            variant = pkl.parent.name
            out_dir = out_root / variant
            t0 = time.time()
            print(f"  [{i:3d}/{len(pkls)}]  {variant}/{pkl.name}")
            _run_single(pkl, out_dir, only=only, force=args.force)
            print(f"      done in {time.time() - t0:.1f}s")
    print(f"\nTotal: {time.time() - overall_t0:.1f}s")
    print(f"Outputs under: {OUT_BASE}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
