"""Run the pipeline on the 180 per-patient × band files of the April batch.

Use ``--subject sub-XXXXXXXX`` to run incrementally on a single subject
while iterating on visualisations / config.

Examples
--------
::

    python scripts/april/03_run_patients.py --out data/correlation_matrices_results/april/patients
    python scripts/april/03_run_patients.py --out ... --subject sub-00246757
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multifunbrain.datasets.april import DEFAULT_APRIL_ROOT, discover_april

from _common import PipelineRunArgs, add_common_args, run_entries  # noqa: E402


class PatientsRunArgs(PipelineRunArgs):
    subject: str | None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_APRIL_ROOT)
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Filter to a single subject (e.g. sub-00246757).",
    )
    add_common_args(parser)
    args = parser.parse_args(argv, namespace=PatientsRunArgs())

    entries = discover_april(args.root, levels=["patient"])
    if args.subject:
        entries = [e for e in entries if e.subject == args.subject]

    return run_entries(entries, args)


if __name__ == "__main__":
    raise SystemExit(main())
