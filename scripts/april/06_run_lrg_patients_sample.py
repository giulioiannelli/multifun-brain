"""Run multiscale LRG on a representative slice of the April per-patient batch.

Selects 2 subjects × 3 bands × 2 contrasts × 1 processing (``bpfBOLD``)
= 12 matrices so the notebook can render patient-level CO2-vs-rest
comparisons without exploding the panel count.

Examples
--------
::

    python scripts/april/06_run_lrg_patients_sample.py \\
        --out data/correlation_matrices_results/april/lrg_patients_sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _lrg_common import LRGRunArgs, add_lrg_args, run_lrg_entries  # noqa: E402

from multifunbrain.datasets.april import DEFAULT_APRIL_ROOT, discover_april

DEFAULT_SUBJECTS = ("sub-00246757", "sub-VA9757")
DEFAULT_PROCESSING = "bpfBOLD"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_APRIL_ROOT)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--processing", default=DEFAULT_PROCESSING)
    add_lrg_args(parser)
    args = parser.parse_args(argv, namespace=LRGRunArgs())

    all_patient = discover_april(args.root, levels=["patient"])
    entries = [
        e for e in all_patient
        if e.subject in args.subjects and e.processing == args.processing
    ]
    return run_lrg_entries(entries, args)


if __name__ == "__main__":
    raise SystemExit(main())
