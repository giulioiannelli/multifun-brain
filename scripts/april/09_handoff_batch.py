"""Collaborator handoff batch driver.

Produces, for every matrix in the April 2026 batch:

- ``fig 1`` (network + LRG descriptive, 4 × 3 grid)
- ``fig 2`` (dendrogram cuts at fixed k ∈ {2, 5, 7, 17, 31})

And, for every pair at the **global** and **inter** (per-band) levels:

- ``fig 3`` (dendrogram comparison + strength-preserving null calibration)
- ``fig3_per_roi_shift.csv`` (per-ROI table)

Pair scope at each level:

- ``global``: 5 within-variant co2-vs-rest + 20 within-contrast
  cross-variant (C(5, 2) × 2) = **25 pairs**.
- ``inter``: same logic **per band** → 25 × 3 = **75 pairs**. No
  cross-band pairs; comparing different bands within a variant is a
  separate question not handled here.

A plain-language ``README.md`` describing methodology + figure captions
is written to the handoff root.

All artefacts are skipped if they already exist; partial caches (surrogate
linkages, pairwise null distributions) are extended in place, so the run
is fully resumable.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from _fig_common import RESULTS_BASE, SRC_BASE, gamma_for_variant, load_atlas
from _handoff_compare import compute_pair
from _handoff_figs import FIG2_K_VALUES, build_fig1, build_fig2
from multifunbrain.datasets.april import AprilEntry, discover_april, load_entry

log = logging.getLogger("handoff")

DEFAULT_OUT_BASE = RESULTS_BASE / "2026-05-21_collaborator-handoff"
DEFAULT_N_SURROGATES = 200

# ────────────────────────────────────────────────────────────────────
# Layout helpers
# ────────────────────────────────────────────────────────────────────


def _matrix_label(e: AprilEntry) -> str:
    """Match the pickle filename stem so cache keys round-trip."""
    if e.level == "global":
        return f"{e.contrast}_{e.processing}_GLOBAL"
    if e.level == "band":
        return f"{e.contrast}_{e.processing}_{e.band}"
    return f"{e.subject}_{e.contrast}_{e.processing}_{e.band}"


def _matrix_out_dir(out_base: Path, e: AprilEntry) -> Path:
    """Resolve the directory that holds fig1 + fig2 for one matrix."""
    if e.level == "global":
        return out_base / "per_matrix" / "global" / e.contrast / e.processing
    if e.level == "band":
        return (
            out_base
            / "per_matrix"
            / "inter"
            / f"{e.band}_{e.contrast}"
            / e.processing
        )
    return (
        out_base
        / "per_matrix"
        / "per_subject"
        / e.subject
        / e.band
        / e.contrast
        / e.processing
    )


def _pair_out_dir(out_base: Path, ea: AprilEntry, eb: AprilEntry) -> Path:
    """Pair-folder layout requested by the collaborator:

    - global:  ``comparisons/global/<labelA>__vs__<labelB>/``
    - inter:   ``comparisons/inter/<band>_<variantA>__vs__<variantB>/``
      (band is a prefix, variant labels carry no band suffix so they
      stay short and recognisable across bands).
    """
    if ea.level == "global":
        return (
            out_base
            / "comparisons"
            / "global"
            / f"{_matrix_label(ea)}__vs__{_matrix_label(eb)}"
        )
    variant_a = f"{ea.contrast}_{ea.processing}"
    variant_b = f"{eb.contrast}_{eb.processing}"
    return (
        out_base
        / "comparisons"
        / "inter"
        / f"{ea.band}_{variant_a}__vs__{variant_b}"
    )


def _enumerate_pairs(
    entries: list[AprilEntry],
) -> list[tuple[str | None, AprilEntry, AprilEntry]]:
    """Within-variant co2-vs-rest + within-contrast cross-variant pairs.

    Returns a list of ``(band_or_None, entry_a, entry_b)`` tuples. For
    global-level entries the band field is None; for band-level entries
    pairs are restricted to within-band combinations.
    """
    groups: dict[str | None, list[AprilEntry]] = defaultdict(list)
    for e in entries:
        groups[e.band].append(e)

    pairs: list[tuple[str | None, AprilEntry, AprilEntry]] = []
    for band, group in groups.items():
        by_key: dict[tuple[str, str], AprilEntry] = {
            (e.contrast, e.processing): e for e in group
        }
        all_proc = sorted({e.processing for e in group})

        for proc in all_proc:
            ea = by_key.get(("co2", proc))
            eb = by_key.get(("rest", proc))
            if ea and eb:
                pairs.append((band, ea, eb))

        for contrast in ("co2", "rest"):
            procs_in_contrast = sorted(
                {e.processing for e in group if e.contrast == contrast}
            )
            for p1, p2 in combinations(procs_in_contrast, 2):
                ea = by_key.get((contrast, p1))
                eb = by_key.get((contrast, p2))
                if ea and eb:
                    pairs.append((band, ea, eb))

    return pairs


# ────────────────────────────────────────────────────────────────────
# Per-matrix / per-pair runners
# ────────────────────────────────────────────────────────────────────


def _build_matrix_figs(
    e: AprilEntry,
    out_base: Path,
    *,
    force: bool,
    runlog: Path,
) -> tuple[int, int]:
    """Return ``(n_built, n_skipped)`` for fig1 + fig2."""
    out_dir = _matrix_out_dir(out_base, e)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = _matrix_label(e)
    gamma = gamma_for_variant(e.processing)

    fig1_path = out_dir / f"{label}_fig1_network_lrg.pdf"
    fig2_path = out_dir / f"{label}_fig2_dendrogram_cuts.pdf"

    if (
        fig1_path.exists()
        and fig2_path.exists()
        and not force
    ):
        return 0, 2

    corr_raw = load_entry(e)

    n_built = 0
    n_skipped = 0
    for fn, path, kind in [
        (build_fig1, fig1_path, "fig1"),
        (build_fig2, fig2_path, "fig2"),
    ]:
        if path.exists() and not force:
            n_skipped += 1
            continue
        t0 = time.time()
        try:
            fn(corr_raw, label, gamma, path)
            dt = time.time() - t0
            print(f"    {kind}  {path.name}  ({dt:.1f}s)")
            with runlog.open("a") as f:
                f.write(
                    f"{label}\t{kind}\tok\t{dt:.1f}s\t{path}\n"
                )
            n_built += 1
        except Exception as exc:
            log.exception("[%s] %s failed: %s", label, kind, exc)
            with runlog.open("a") as f:
                f.write(f"{label}\t{kind}\tERR\t{type(exc).__name__}: {exc}\n")
    return n_built, n_skipped


def _build_pair_fig(
    band: str | None,
    ea: AprilEntry,
    eb: AprilEntry,
    out_base: Path,
    *,
    n_surrogates: int,
    cache_dir: Path,
    atlas,
    force: bool,
    runlog: Path,
    null_kind: str,
    rng_seed: int,
) -> tuple[int, int]:
    label_a = _matrix_label(ea)
    label_b = _matrix_label(eb)
    pair_dir = _pair_out_dir(out_base, ea, eb)
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_key = pair_dir.name
    out_pdf = pair_dir / "fig3_compare.pdf"
    out_csv = pair_dir / "fig3_per_roi_shift.csv"

    if out_pdf.exists() and out_csv.exists() and not force:
        return 0, 1

    corr_a = load_entry(ea)
    corr_b = load_entry(eb)
    gamma_a = gamma_for_variant(ea.processing)
    gamma_b = gamma_for_variant(eb.processing)
    atlas_img, atlas_data, atlas_labels = atlas

    t0 = time.time()
    try:
        summary = compute_pair(
            corr_a, label_a, corr_b, label_b,
            gamma_a=gamma_a, gamma_b=gamma_b,
            atlas_img=atlas_img, atlas_data=atlas_data,
            atlas_labels=atlas_labels,
            out_pdf=out_pdf, out_csv=out_csv,
            null_kind=null_kind,
            n_surrogates=n_surrogates,
            cache_dir=cache_dir,
            rng_seed=rng_seed,
        )
        dt = time.time() - t0
        print(
            f"    pair  {pair_key}  ({dt:.1f}s)  "
            f"median_obs={summary.get('median_shift', float('nan')):.3f}  "
            f"median_calib={summary.get('median_calibrated', float('nan')):.3f}"
        )
        with runlog.open("a") as f:
            f.write(
                f"{pair_key}\tpair\tok\t{dt:.1f}s\t"
                f"median_obs={summary.get('median_shift'):.4f}\t"
                f"median_calib={summary.get('median_calibrated'):.4f}\n"
            )
        return 1, 0
    except Exception as exc:
        log.exception("[%s] pair failed: %s", pair_key, exc)
        with runlog.open("a") as f:
            f.write(f"{pair_key}\tpair\tERR\t{type(exc).__name__}: {exc}\n")
        return 0, 0


# ────────────────────────────────────────────────────────────────────
# README writer
# ────────────────────────────────────────────────────────────────────


README_TEMPLATE = """\
# {date} — multifun-brain handoff

This folder contains the LRG-based analysis of the April 2026 fMRI
correlation matrices, in a single place, ready to read.

There are **two figures per correlation matrix** and **one comparison
figure per pair of matrices**:

- `fig 1` — what the matrix looks like before and after cleaning, plus
  the LRG hierarchy summary at the most informative diffusion scale.
- `fig 2` — the same hierarchy, **cut at five different granularities**
  (k = 2, 5, 7, 17, 31 clusters) so cross-matrix comparison is fair.
- `fig 3` — for each pair, *which brain regions reorganise most* when
  the network is rebuilt from the other matrix, plus a principled noise
  baseline.

Every figure exists under `per_matrix/` (one figure per condition) or
`comparisons/` (one figure per pair).

The numbers behind `fig 3` live next to it as a CSV (`fig3_per_roi_shift.csv`)
sorted by reorganisation score.

---

## How the matrices were cleaned

The collaborator pickles are 100 × 100 Schaefer-2018 correlation matrices
(17-network parcellation). Three cleaning steps, all standard, all
automated:

1. **Dead-region removal.** If a parcel has no valid timeseries (all NaN
   rows / columns in the correlation matrix), it is *dropped*, not
   zero-filled. Zero-filling would invent data and bias diffusion
   results. Dropped indices are listed in the per-matrix CSVs.

2. **Marchenko–Pastur (MP) denoising.** Random-matrix theory tells us
   that with `n_timepoints` time points and `p` parcels, even pure
   noise produces eigenvalues up to the **MP edge** `(1 + √γ)²`,
   `γ = p / n_timepoints`. We subtract everything below that edge so
   only structured covariance remains. The γ used for each processing
   variant is shown on the corresponding figure.

3. **LANS backbone.** From the MP-cleaned positive-correlation matrix
   we keep only edges that are statistically *locally significant* at
   α = 0.05 (Serrano-Boguñá disparity filter, variant LANS). The result
   is a sparse weighted network where every link is non-trivial.

The LRG hierarchy is built on this LANS backbone.

---

## What LRG does (one paragraph)

The Laplacian Renormalisation Group treats the brain network like a
substrate on which a *heat pulse* spreads. The pulse is governed by the
graph Laplacian; small time τ keeps it local, large τ lets it mix
everywhere. At the **smallest informative scale**, τ_min = 1 / λ_max
(the inverse of the largest Laplacian eigenvalue), pairs of regions
that the pulse cannot bridge in that time look "far apart" — and pairs
it does bridge look "close". Hierarchical clustering on those diffusion
distances gives the dendrogram you see on every figure. Cutting the
dendrogram at different heights yields partitions with different
numbers of clusters (k); we show k = 2, 5, 7, 17, 31 for comparison.

---

## How to read `fig 3` — the comparison figure

### The raw per-ROI shift (what we measure before any null)

For each parcel `i` we look at its row in the cophenetic-distance
matrix of tree A (CO2, say) and the matching row in tree B (rest). We
**rank-transform** the off-diagonal pairs separately inside each tree
(so the numbers live on the same scale regardless of how the merge
heights stretch), then average `|rank_A(i, j) − rank_B(i, j)|` over all
partners `j ≠ i`, and divide by the number of pairs so it falls in
**[0, 1]**:

| raw shift | meaning |
|---|---|
| **0** | parcel `i` sits next to the same neighbours, in the same order, in both trees |
| **1** | maximum possible disagreement on this parcel's neighbours |
| **~0.3** | the value two *unrelated* trees of a 100-leaf network produce on average — this is the **noise floor** |

So a raw shift of 0.3 alone tells you nothing about whether two
conditions really differ — that's just what two random trees of this
size give. We have to *calibrate*.

### The strength-preserving null

To know how much disagreement random noise alone would produce, we
generate surrogate networks:

> Take the LANS backbone, **shuffle its edges** while keeping each
> region's total connection weight identical. Build the LRG hierarchy
> on the shuffled network. Compare to the LRG hierarchy of the *other*
> condition shuffled the same way. Repeat 200 times.

For every surrogate pair we recompute the per-ROI shift. Per parcel we
then have ~200 null values, and we take their **median** as that
parcel's noise floor. Calibration is per-parcel, not pooled:

```
calibrated_shift[i] = observed_shift[i] − null_median[i]
```

The brain maps in `fig 3` colour each parcel by `calibrated_shift`.

### Reading the colours

| calibrated value | meaning |
|---|---|
| **0** | this parcel disagrees between A and B exactly as much as strength-preserving rewiring would predict for it. **No detectable reorganisation.** |
| **+0.1** | this parcel disagrees 0.1 units **more** than the noise floor. Genuine reorganisation under the contrast. |
| **−0.1** | this parcel disagrees 0.1 units **less** than the noise floor. **More locked-in than chance** — the two conditions agree on this parcel's position more than random strength-matched trees would. |

A grey parcel is *not* "nothing happens" — it's "at the noise floor".
The interesting parcels are the saturated reds (real rewiring) **and**
the deep blues (held in place beyond chance).

### The two scalars in the title — Baker's γ and cophenetic r

These are *tree-level* numbers, one per pair, both in `[-1, +1]`:

- **Baker's γ** — Kendall-τ on cophenetic-distance **ranks**: "do the
  two trees merge things in the same *order*?" +1 = identical merge
  order, 0 = no rank agreement, −1 = reversed.
- **Cophenetic r** — Pearson on cophenetic-distance **values**: "do the
  merge *heights* agree linearly?" Same scale, more sensitive to
  magnitudes.

Read them as a **global summary**: γ ≈ 0 means the two dendrograms are
uncorrelated as a whole. That's not a contradiction with positive
per-parcel calibrated shifts — globally the trees can look like random
rewirings of each other while a handful of specific parcels still carry
a real signal above noise. The per-parcel map is what tells you *which*
regions.

### The bottom-left histogram

Pooled observed vs pooled null shifts across all parcels — purely a
visual sanity check. If the red (observed) bar mass sits to the right
of the grey (null) bar mass, there's a small systematic effect at the
pair level on top of the per-ROI signal.

---

## Folder layout

```
{root_name}/
├── README.md                                    (this file)
├── per_matrix/
│   ├── global/<contrast>/<variant>/             (fig1 + fig2)
│   ├── inter/<band>_<contrast>/<variant>/
│   └── per_subject/<subject>/<band>/<contrast>/<variant>/
├── comparisons/
│   ├── global/<labelA>__vs__<labelB>/           (fig3 + per-ROI CSV)
│   └── inter/<band>_<variantA>__vs__<variantB>/
└── _cache/                                      (surrogate trees, null
                                                  distributions, run log)
```

Counts in this build: {n_global_mat} global matrices, {n_inter_mat}
inter (per-band) matrices, {n_subject_mat} per-subject matrices,
{n_global_pairs} global pairs, {n_inter_pairs} inter pairs.

---

## Figure captions

### `fig 1` — network + LRG descriptive (4 × 3 grid)

Top three rows show the cleaning pipeline:

- **Row 0**: raw signed correlation heatmap | edge-weight distribution
  (raw negative red, raw positive blue, MP-clean overlay dark grey) |
  MP-denoised correlation heatmap.
- **Row 1**: filter density bar chart (how many edges each filter
  keeps) | MP-clean signed network drawn on a force layout | the
  MP-clean ∩ (weight > 0) matrix used downstream.
- **Row 2**: percolation curve (largest component vs threshold) | node
  strength distribution | weighted clustering coefficient distribution.

Bottom row shows the LRG hierarchy at τ_min:

- **Row 3**: LANS backbone coloured by community at τ_min (largest
  community = palette[0]) | dendrogram on log-y, cut shown at the most
  stable height | partition stability index Ψ as a function of k, with
  the optimal k highlighted in red.

### `fig 2` — dendrogram cuts at fixed k (5 × 3 grid)

One row per k ∈ {{2, 5, 7, 17, 31}}. Cluster colours propagate from the
coarsest cut outward (largest groups keep their palette slot as they
split). Columns:

- **Network** — LANS backbone with the k-partition.
- **Brain** — Schaefer parcels coloured the same way, ortho slices.
- **Dendrogram** — log-y, horizontal cut line at the height that yields k.

Cross-matrix comparison: pick a k, compare the corresponding brain
panel from any two matrices.

### `fig 3` — pairwise comparison

For one pair of matrices, three brain-orientation montages (axial,
sagittal, coronal) coloured by **calibrated** per-ROI reorganisation
shift (positive = reorganised beyond noise; negative = even more
stable than the noise floor). Bottom row: observed-vs-null histogram
(left), top-10 reorganised + top-10 stable parcels with names (right).
The numbers behind these panels live in `fig3_per_roi_shift.csv` —
columns:

- `atlas_index` — Schaefer 0-based index (0–99).
- `roi_name` — short Yeo-network label.
- `shift_score` — raw cophenetic-rank shift in [0, 1].
- `null_median`, `null_p95` — strength-preserving null statistics.
- `null_n_samples` — number of surrogate pairs contributing.
- `calibrated_shift` — observed − null median.

---

Generated by `scripts/april/09_handoff_batch.py`.
"""


def _write_readme(
    out_base: Path,
    *,
    counts: dict[str, int],
    date_str: str,
) -> None:
    body = README_TEMPLATE.format(
        date=date_str,
        root_name=out_base.name,
        n_global_mat=counts.get("global", 0),
        n_inter_mat=counts.get("inter", 0),
        n_subject_mat=counts.get("subject", 0),
        n_global_pairs=counts.get("global_pairs", 0),
        n_inter_pairs=counts.get("inter_pairs", 0),
    )
    out_base.mkdir(parents=True, exist_ok=True)
    with (out_base / "README.md").open("w") as f:
        f.write(body)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_BASE,
        help="Output root for the handoff folder.",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=DEFAULT_N_SURROGATES,
        help="Number of strength-preserving surrogates per matrix.",
    )
    parser.add_argument(
        "--levels", default="global,inter,per_subject",
        help="Comma-separated subset of {global,inter,per_subject}. "
        "Default: all three.",
    )
    parser.add_argument(
        "--null", choices=["strength-preserving", "subject-split", "none"],
        default="strength-preserving",
        help="Null model for fig 3 (default: strength-preserving).",
    )
    parser.add_argument(
        "--null-seed", type=int, default=0,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-render existing artefacts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be produced; build nothing.",
    )
    parser.add_argument(
        "--date", default="2026-05-21",
        help="Date stamp for the README header.",
    )
    args = parser.parse_args(argv)

    out_base: Path = args.out_dir
    cache_dir = out_base / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runlog = cache_dir / "run.log"
    runlog.touch(exist_ok=True)

    levels_req = {s.strip() for s in args.levels.split(",") if s.strip()}
    level_map = {"global": "global", "inter": "band", "per_subject": "patient"}
    levels_lookup = [level_map[L] for L in levels_req if L in level_map]
    if not levels_lookup:
        sys.exit("No valid --levels.")

    print(f"Discovering April entries (levels = {sorted(levels_req)}) ...")
    entries = discover_april(SRC_BASE, levels=levels_lookup)
    by_level: dict[str, list[AprilEntry]] = defaultdict(list)
    for e in entries:
        by_level[e.level].append(e)

    counts = {
        "global": len(by_level.get("global", [])),
        "inter": len(by_level.get("band", [])),
        "subject": len(by_level.get("patient", [])),
    }

    pairs_by_level: dict[str, list[tuple[str | None, AprilEntry, AprilEntry]]] = {}
    if "global" in levels_req:
        pairs_by_level["global"] = _enumerate_pairs(by_level.get("global", []))
    if "inter" in levels_req:
        pairs_by_level["inter"] = _enumerate_pairs(by_level.get("band", []))

    counts["global_pairs"] = len(pairs_by_level.get("global", []))
    counts["inter_pairs"] = len(pairs_by_level.get("inter", []))

    print(
        "Matrices: "
        + ", ".join(f"{k} = {v}" for k, v in counts.items() if "pairs" not in k)
    )
    print(
        "Pairs: "
        + ", ".join(f"{k} = {v}" for k, v in counts.items() if "pairs" in k)
    )

    if args.dry_run:
        for level, group in by_level.items():
            print(f"\n[dry-run] level={level} ({len(group)} matrices)")
            for e in group[:5]:
                print(f"  {_matrix_label(e)}  →  {_matrix_out_dir(out_base, e)}")
            if len(group) > 5:
                print(f"  ... and {len(group) - 5} more")
        for level, plist in pairs_by_level.items():
            print(f"\n[dry-run] pairs {level} ({len(plist)})")
            for band, ea, eb in plist[:5]:
                key = f"{_matrix_label(ea)}__vs__{_matrix_label(eb)}"
                print(f"  {key}  →  {_pair_out_dir(out_base, ea, eb)}")
            if len(plist) > 5:
                print(f"  ... and {len(plist) - 5} more")
        _write_readme(out_base, counts=counts, date_str=args.date)
        print(f"README → {out_base / 'README.md'}")
        return 0

    # ── matrix figures ─────────────────────────────────────────────────
    n_built_total = 0
    n_skipped_total = 0
    for level in ("global", "band", "patient"):
        group = by_level.get(level, [])
        if not group:
            continue
        print(f"\n=== matrices: {level} ({len(group)}) ===")
        for i, e in enumerate(group, 1):
            label = _matrix_label(e)
            print(f"  [{i:3d}/{len(group)}] {label}")
            built, skipped = _build_matrix_figs(
                e, out_base, force=args.force, runlog=runlog,
            )
            n_built_total += built
            n_skipped_total += skipped

    # ── pair figures (null cache) ──────────────────────────────────────
    atlas = load_atlas()
    for level in ("global", "inter"):
        plist = pairs_by_level.get(level, [])
        if not plist:
            continue
        print(f"\n=== pairs: {level} ({len(plist)}) ===")
        for i, (band, ea, eb) in enumerate(plist, 1):
            pair_key = f"{_matrix_label(ea)}__vs__{_matrix_label(eb)}"
            print(f"  [{i:3d}/{len(plist)}] {pair_key}")
            built, skipped = _build_pair_fig(
                band, ea, eb, out_base,
                n_surrogates=args.n_surrogates,
                cache_dir=cache_dir,
                atlas=atlas,
                force=args.force,
                runlog=runlog,
                null_kind=args.null,
                rng_seed=args.null_seed,
            )
            n_built_total += built
            n_skipped_total += skipped

    _write_readme(out_base, counts=counts, date_str=args.date)
    print(
        f"\nDone. Built {n_built_total} new artefacts, "
        f"skipped {n_skipped_total} pre-existing.\n"
        f"Handoff folder: {out_base}\n"
        f"README: {out_base / 'README.md'}\n"
        f"Run log: {runlog}"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
