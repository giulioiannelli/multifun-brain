"""CLI wrapper for the pairwise comparison figure (``fig 3``).

Outputs:

- ``<out_dir>/fig3_compare.pdf`` — 3 brain-orientation montages + null
  histogram + top/bottom ROI bar chart.
- ``<out_dir>/fig3_per_roi_shift.csv`` — per-ROI table with raw and
  calibrated shift, null median / p95.

Default null: **strength-preserving** — see ``_handoff_compare`` for the
algorithm. ``--null subject-split`` and ``--null none`` are kept as
alternatives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _fig_common import RESULTS_BASE, SRC_BASE, gamma_for_variant, load_atlas
from _handoff_compare import compute_pair
from multifunbrain.io.corrmatrix import load_correlation_matrix

OUT_BASE = RESULTS_BASE / "_compare_figs"


def _resolve_label(label: str) -> tuple[Path, str]:
    variant = label.rsplit("_", 1)[0]
    suffix = label.rsplit("_", 1)[1]
    sub = "freq-contrast-global" if suffix == "GLOBAL" else "freq-contrast-inter"
    return SRC_BASE / sub / variant / f"{label}.pkl", variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        default="co2_optcom_bold_GLOBAL,rest_optcom_bold_GLOBAL",
        help="Two pickle labels (no .pkl), comma-separated. Looked up "
        "under SRC_BASE/freq-contrast-global/<variant>/<label>.pkl",
    )
    parser.add_argument("--out-dir", default=None, help="Override OUT_BASE.")
    parser.add_argument(
        "--null",
        choices=["strength-preserving", "subject-split", "none"],
        default="strength-preserving",
        help="Null model (default: strength-preserving).",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=200,
        help="Number of surrogates for strength-preserving null.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache directory for surrogate and pair null pickles. "
        "Default: <out-dir>/_cache.",
    )
    parser.add_argument("--null-seed", type=int, default=0)
    args = parser.parse_args()

    label_a, label_b = [s.strip() for s in args.pair.split(",")]
    path_a, var_a = _resolve_label(label_a)
    path_b, var_b = _resolve_label(label_b)
    if not path_a.exists():
        sys.exit(f"missing: {path_a}")
    if not path_b.exists():
        sys.exit(f"missing: {path_b}")
    if var_a != var_b:
        print(f"NOTE: variants differ ({var_a} vs {var_b}); using per-side gamma.")

    gamma_a = gamma_for_variant(var_a)
    gamma_b = gamma_for_variant(var_b)
    corr_a = load_correlation_matrix(path_a)
    corr_b = load_correlation_matrix(path_b)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_BASE / f"{label_a}__vs__{label_b}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig3_compare.pdf"
    out_csv = out_dir / "fig3_per_roi_shift.csv"

    cache_dir = args.cache_dir or (out_dir.parent / "_cache")

    atlas_img, atlas_data, atlas_labels = load_atlas()

    print(f"=== fig 3: {label_a}  vs  {label_b}  (null = {args.null}) ===")
    summary = compute_pair(
        corr_a, label_a, corr_b, label_b,
        gamma_a=gamma_a, gamma_b=gamma_b,
        atlas_img=atlas_img, atlas_data=atlas_data,
        atlas_labels=atlas_labels,
        out_pdf=out_pdf, out_csv=out_csv,
        null_kind=args.null,
        n_surrogates=args.n_surrogates,
        cache_dir=cache_dir,
        rng_seed=args.null_seed,
    )
    print(f"  PDF: {out_pdf}")
    print(f"  CSV: {out_csv}")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
