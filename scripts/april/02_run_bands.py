"""Run the pipeline on the 30 per-band aggregates of the April batch.

Examples
--------
::

    python scripts/april/02_run_bands.py --out data/correlation_matrices_results/april/bands
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multifunbrain.datasets.april import DEFAULT_APRIL_ROOT, discover_april

from _common import PipelineRunArgs, add_common_args, run_entries  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_APRIL_ROOT)
    add_common_args(parser)
    args = parser.parse_args(argv, namespace=PipelineRunArgs())

    entries = discover_april(args.root, levels=["band"])
    return run_entries(entries, args)


if __name__ == "__main__":
    raise SystemExit(main())
