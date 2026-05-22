"""Run multiscale LRG on the 10 global aggregates of the April batch.

Both ``mp_validated`` (canonical) and ``percolation`` (sensitivity)
backbones are run per matrix. Output layout::

    <out>/<backbone>/global/<contrast>_<processing>.pkl
    <out>/manifest.csv
    <out>/config.json
    <out>/failed_matrices.json

Examples
--------
::

    python scripts/april/04_run_lrg_global.py \\
        --out data/correlation_matrices_results/april/lrg_global
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

    entries = discover_april(args.root, levels=["global"])
    return run_lrg_entries(entries, args)


if __name__ == "__main__":
    raise SystemExit(main())
