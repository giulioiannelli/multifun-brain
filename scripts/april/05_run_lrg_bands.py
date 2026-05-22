"""Run multiscale LRG on the 30 per-band aggregates of the April batch.

Each entry is one of (2 contrasts × 5 processing × 3 bands) = 30 matrices,
aggregated across the 6 subjects. Per-band gamma matches the parent
processing variant.

Examples
--------
::

    python scripts/april/05_run_lrg_bands.py \\
        --out data/correlation_matrices_results/april/lrg_bands
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _lrg_common import LRGRunArgs, add_lrg_args, run_lrg_entries  # noqa: E402

from multifunbrain.datasets.april import DEFAULT_APRIL_ROOT, discover_april


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_APRIL_ROOT)
    add_lrg_args(parser)
    args = parser.parse_args(argv, namespace=LRGRunArgs())

    entries = discover_april(args.root, levels=["band"])
    return run_lrg_entries(entries, args)


if __name__ == "__main__":
    raise SystemExit(main())
