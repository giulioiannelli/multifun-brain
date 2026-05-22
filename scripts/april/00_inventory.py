"""Build a manifest CSV for the April correlation-matrix batch.

Walks the metadata-tagged loader and writes a CSV to
``data/correlation_matrices_results/april/manifest.csv`` (by default)
with one row per discovered correlation matrix.

Examples
--------
::

    python scripts/april/00_inventory.py
    python scripts/april/00_inventory.py --root path/to/data --out custom/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multifunbrain.datasets.april import (
    DEFAULT_APRIL_ROOT,
    discover_april,
    entries_to_dataframe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_APRIL_ROOT,
        help="Root directory of the April batch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/correlation_matrices_results/april/manifest.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    entries = discover_april(args.root)
    if not entries:
        print(f"No entries discovered under {args.root}", file=sys.stderr)
        return 1

    df = entries_to_dataframe(entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    counts = df.groupby("level").size().to_dict()
    print(f"Discovered {len(df)} matrices → {args.out}")
    for level, n in sorted(counts.items()):
        print(f"  {level}: {n}")
    print(f"  contrasts: {sorted(df['contrast'].unique().tolist())}")
    print(f"  processing variants: {sorted(df['processing'].unique().tolist())}")
    print(f"  bands: {sorted(b for b in df['band'].unique().tolist() if b)}")
    subjects = sorted(s for s in df["subject"].unique().tolist() if s)
    print(f"  subjects: {len(subjects)} ({subjects})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
