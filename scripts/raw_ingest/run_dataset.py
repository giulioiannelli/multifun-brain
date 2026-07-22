"""Ingest one raw dataset into dashboard result bundles (global/bands/patients).

    # by dataset id: data/raw_data_<id> -> data/correlation_matrices_results/<id>
    python -m scripts.raw_ingest.run_dataset --dataset schaefer100_november2025
    python -m scripts.raw_ingest.run_dataset --dataset harvardoxford48

    # explicit paths
    python -m scripts.raw_ingest.run_dataset \
        --raw-root data/raw_data_harvardoxford48 \
        --out      data/correlation_matrices_results/harvardoxford48

The heavy step is EMD over every ROI plus the LRG per matrix, so a full dataset
takes a while — run it detached. Degenerate high-frequency bands (above a slow
acquisition's Nyquist limit) are recorded in ``failed_matrices.json``, not fatal.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from multifunbrain import PipelineConfig

from ._common import dump_bundle, ingest_dataset, plan_size


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", help="raw dataset id (resolves raw + out under --data-root)")
    ap.add_argument("--raw-root", type=Path, help="explicit raw_data_<id> directory")
    ap.add_argument("--out", type=Path, help="explicit output bundle root")
    ap.add_argument(
        "--data-root", type=Path, default=Path("data"), help="base data dir (default: data)"
    )
    args = ap.parse_args()

    if args.dataset:
        raw_root = args.raw_root or args.data_root / f"raw_data_{args.dataset}"
        out_root = args.out or args.data_root / "correlation_matrices_results" / args.dataset
    elif args.raw_root and args.out:
        raw_root, out_root = args.raw_root, args.out
    else:
        ap.error("give --dataset, or both --raw-root and --out")

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    total = plan_size(raw_root)
    print(f"[ingest] {raw_root} -> {out_root}  ({total} pipeline runs planned)", flush=True)

    t0 = time.monotonic()
    done = 0

    def progress(label: str) -> None:
        nonlocal done
        done += 1
        if done % 20 == 0 or done == total:
            el = time.monotonic() - t0
            rate = done / el if el else 0.0
            eta = (total - done) / rate if rate else 0.0
            print(
                f"[ingest] {done}/{total}  {el / 60:.1f} min elapsed, "
                f"~{eta / 60:.1f} min left  (last: {label})",
                flush=True,
            )

    bundles = ingest_dataset(raw_root, progress=progress)

    cfg = PipelineConfig()  # reference snapshot for config.json (per-run gamma differs)
    for name, results in bundles.items():
        dump_bundle(out_root / name, results, cfg)
        n_ok = sum(1 for r in results if r.error is None)
        print(f"[ingest] {name}: {n_ok}/{len(results)} ok -> {out_root / name}", flush=True)

    print(f"[ingest] DONE {raw_root.name} in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
