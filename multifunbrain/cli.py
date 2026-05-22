"""Command-line interface for :mod:`multifunbrain`."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from .generation import generate_hmn
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    discover_matrices,
    load_results,
    run_pipeline,
)
from .visualization import (
    plot_correlation_matrix,
    plot_filtered_comparison,
    plot_lrg_dendrogram,
    plot_lrg_entropy,
    plot_lrg_partition_network,
    plot_lrg_psi,
    plot_lrg_sankey,
    plot_network,
    plot_node_metrics,
    plot_pipeline_summary,
    plot_signed_balance,
    plot_signed_laplacian_spectrum,
    plot_signed_network,
    plot_weight_distribution,
)

log = logging.getLogger(__name__)

CommandHandler = Callable[[argparse.Namespace], int]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multifunbrain",
        description="Portable utilities for hierarchical modular brain networks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hello_parser = subparsers.add_parser("hello", help="Print a friendly greeting")
    hello_parser.add_argument(
        "name",
        nargs="?",
        default="Researcher",
        help="Name to greet (default: Researcher)",
    )
    hello_parser.set_defaults(func=_cmd_hello)

    gen_parser = subparsers.add_parser(
        "generate-hmn",
        help="Generate a hierarchical modular network and print summary statistics",
    )
    gen_parser.add_argument("--levels", type=int, default=3, help="Hierarchy depth")
    gen_parser.add_argument(
        "--base-module-size",
        type=int,
        default=4,
        help="Number of nodes per base module",
    )
    gen_parser.add_argument(
        "--p-in",
        type=float,
        default=0.9,
        help="Intra-module connection probability",
    )
    gen_parser.add_argument(
        "--p-out",
        type=float,
        default=0.05,
        help="Inter-module connection probability",
    )
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the graph as GraphML",
    )
    gen_parser.set_defaults(func=_cmd_generate_hmn)

    # ── analyze subcommand ────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run the correlation network analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the three-section pipeline on one or more correlation matrices.\n\n"
            "Examples:\n"
            "  # Single file\n"
            "  multifunbrain analyze data/correlation_matrices/my_matrix.pkl\n\n"
            "  # All matrices in a directory (recursive)\n"
            "  multifunbrain analyze data/correlation_matrices/\n\n"
            "  # Only Bold matrices, skip LRG\n"
            "  multifunbrain analyze data/correlation_matrices/ --pattern Bold --no-lrg\n\n"
            "  # With RMT validation (gamma = n_regions / n_timepoints)\n"
            "  multifunbrain analyze data/correlation_matrices/ --gamma 0.19\n\n"
            "  # Save results to a specific directory\n"
            "  multifunbrain analyze data/correlation_matrices/ -o results/"
        ),
    )
    analyze_parser.add_argument(
        "input",
        type=Path,
        help="A single matrix file (.npy/.npz/.csv/.txt/.pkl) or a directory",
    )
    analyze_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: <input>_results/)",
    )
    analyze_parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Aspect ratio p/n for RMT (n_regions / n_timepoints)",
    )
    analyze_parser.add_argument(
        "--filters",
        type=str,
        nargs="+",
        default=["absolute", "positive", "negative"],
        help="Filtering methods (default: absolute positive negative)",
    )
    analyze_parser.add_argument(
        "--no-lrg",
        action="store_true",
        help="Skip LRG multiscale analysis (faster)",
    )
    analyze_parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip standard network metrics (only run descriptive + filtering)",
    )
    analyze_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Only include files whose name contains PATTERN (case-insensitive)",
    )
    analyze_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories",
    )
    analyze_parser.add_argument(
        "--precision-method",
        type=str,
        choices=["direct", "orie", "graphical_lasso"],
        default="direct",
        help="Precision matrix estimation method (default: direct)",
    )
    analyze_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    analyze_parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered files, don't run the pipeline",
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed logging (default: quiet, progress bar only)",
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    # ── plot subcommand ──────────────────────────────────────────────
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots from saved pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Generate visualisation plots from saved pipeline results.\n\n"
            "Examples:\n"
            "  # All plots for all results\n"
            "  multifunbrain plot pipeline_results/\n\n"
            "  # Only Bold matrices\n"
            "  multifunbrain plot pipeline_results/ --pattern Bold\n\n"
            "  # Only 6-panel summaries\n"
            "  multifunbrain plot pipeline_results/ --plots summary\n\n"
            "  # PDF output at high resolution\n"
            "  multifunbrain plot pipeline_results/ -o figures/ --format pdf --dpi 300"
        ),
    )
    plot_parser.add_argument(
        "input",
        type=Path,
        help="Results directory or results.pkl file from a previous analyze run",
    )
    plot_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: <input>/figures/)",
    )
    plot_parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Which plots to generate (default: all). Choices: "
            "summary, correlation_matrix, weight_distribution, "
            "signed_balance, signed_network, signed_laplacian, "
            "filtered_comparison, network, node_metrics, "
            "lrg_entropy, lrg_dendrogram, lrg_psi, lrg_partition, lrg_sankey"
        ),
    )
    plot_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Only plot results whose label contains PATTERN (case-insensitive)",
    )
    plot_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        dest="fmt",
        help="Output image format (default: png)",
    )
    plot_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution in dots per inch (default: 150)",
    )
    plot_parser.set_defaults(func=_cmd_plot)

    return parser


def _cmd_hello(args: argparse.Namespace) -> int:
    print(f"Hello, {args.name}! Welcome to multifun-brain.")
    return 0


def _cmd_generate_hmn(args: argparse.Namespace) -> int:
    graph = generate_hmn(
        levels=args.levels,
        base_module_size=args.base_module_size,
        p_in=args.p_in,
        p_out=args.p_out,
        seed=args.seed,
    )

    n_nodes = graph.number_of_nodes()
    degrees = dict(graph.degree()).values()
    average_degree = float(sum(degrees)) / n_nodes if n_nodes else 0.0

    stats: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "average_degree": average_degree,
        "levels": args.levels,
        "base_module_size": args.base_module_size,
        "p_in": args.p_in,
        "p_out": args.p_out,
        "seed": args.seed,
    }
    print(json.dumps(stats, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, args.output)
        print(f"GraphML written to {args.output}", file=sys.stderr)

    return 0


def _save_result_json(result, json_dir: Path) -> None:
    """Save a single PipelineResult as JSON (progressive save)."""
    rel = Path(result.label or "unlabeled")
    json_path = json_dir / rel.parent / f"{rel.name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def _cmd_analyze(args: argparse.Namespace) -> int:
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        gamma=args.gamma,
        precision_method=args.precision_method,
        filter_methods=args.filters,
        run_lrg=not args.no_lrg,
        run_standard_metrics=not args.no_metrics,
        run_community_detection=not args.no_metrics,
        seed=args.seed,
    )

    input_path: Path = args.input.resolve()

    # ── Resolve output directory early (needed for progressive save) ──
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif input_path.is_file():
        output_dir = input_path.parent / (input_path.stem + "_results")
    else:
        output_dir = input_path.parent / (input_path.name + "_results")
    output_dir = output_dir.resolve()

    # ── Single file ───────────────────────────────────────────────────
    if input_path.is_file():
        if args.list_only:
            print(input_path)
            return 0
        print(f"Running pipeline on: {input_path.name}", file=sys.stderr)
        files = [input_path]
        labels = [None]

    # ── Directory ─────────────────────────────────────────────────────
    elif input_path.is_dir():
        recursive = not args.no_recursive
        files = discover_matrices(
            input_path,
            recursive=recursive,
            pattern=args.pattern,
        )
        if not files:
            print("No matrix files found.", file=sys.stderr)
            return 1

        if args.list_only:
            for f in files:
                print(f)
            print(f"\n{len(files)} files found.", file=sys.stderr)
            return 0

        from .pipeline.discovery import label_from_path
        labels = [label_from_path(f, input_path) for f in files]
        print(f"Found {len(files)} matrices. Running pipeline...", file=sys.stderr)
    else:
        print(f"Error: {input_path} is not a file or directory.", file=sys.stderr)
        return 1

    # ── Prepare output dirs ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "per_matrix"
    json_dir.mkdir(exist_ok=True)

    # ── Process matrices with progressive save + progress bar ────────
    import warnings

    from tqdm import tqdm

    # Send ALL logging + Python warnings to a log file so the progress
    # bar on stderr stays clean.  With --verbose they also go to stderr.
    log_path = output_dir / "analysis.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    )

    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers = [file_handler]
    root_logger.setLevel(logging.DEBUG)

    # Also redirect Python warnings (RuntimeWarning etc.) to the log file.
    logging.captureWarnings(True)

    # If --verbose, additionally print to stderr through tqdm.write.
    if args.verbose:
        class _TqdmHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    tqdm.write(self.format(record), file=sys.stderr)
                except Exception:
                    self.handleError(record)

        _tqdm_h = _TqdmHandler()
        _tqdm_h.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
        )
        root_logger.addHandler(_tqdm_h)

    results: list = []
    failed: list[dict[str, str]] = []

    pbar = tqdm(
        zip(files, labels),
        total=len(files),
        desc="Analyzing",
        unit="matrix",
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for fpath, label in pbar:
        short = label or fpath.name
        pbar.set_postfix_str(short, refresh=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                r = run_pipeline(fpath, config=config, label=label)
        except Exception as exc:
            log.error("Pipeline crashed for '%s': %s", short, exc)
            r = PipelineResult(
                config=config,
                label=label,
                error=f"Unexpected error: {exc}",
            )

        if r.error:
            failed.append({"label": short, "reason": r.error})

        results.append(r)

        # Save JSON immediately so partial results survive crashes.
        _save_result_json(r, json_dir)

    pbar.close()

    # Restore original log handlers.
    logging.captureWarnings(False)
    file_handler.close()
    root_logger.handlers = original_handlers

    # ── Final aggregated saves ───────────────────────────────────────
    # Summary CSV
    summary_frames = [r.summary_table() for r in results if len(r.network_analyses) > 0]
    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        csv_path = output_dir / "summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Summary table saved to {csv_path}", file=sys.stderr)
        print(summary.to_string(index=False))

    # Full results as pickle
    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nFull results saved to {pkl_path}", file=sys.stderr)

    print(f"Per-matrix JSON summaries in {json_dir}/", file=sys.stderr)

    # ── Report skipped / failed matrices ─────────────────────────────
    n_ok = len(results) - len(failed)
    print(f"\nDone. {n_ok}/{len(results)} matrices analyzed successfully.",
          file=sys.stderr)

    if failed:
        # Write a failures report file
        report_path = output_dir / "failed_matrices.json"
        with open(report_path, "w") as f:
            json.dump(failed, f, indent=2)

        print(f"\n{len(failed)} matrix(es) skipped or failed:", file=sys.stderr)
        for entry in failed:
            print(f"  - {entry['label']}: {entry['reason']}", file=sys.stderr)
        print(f"\nFull failure report saved to {report_path}", file=sys.stderr)

    return 0


# ── Plot registries ──────────────────────────────────────────────────
# Per-result plots: one figure per matrix (no filter dimension).
_PLOTS_PER_RESULT: dict[str, Callable] = {
    "summary": plot_pipeline_summary,
    "correlation_matrix": plot_correlation_matrix,
    "weight_distribution": plot_weight_distribution,
    "signed_balance": plot_signed_balance,
    "signed_network": plot_signed_network,
    "filtered_comparison": plot_filtered_comparison,
}

# Per-filter plots: one figure per filter per matrix.
_PLOTS_PER_FILTER: dict[str, Callable] = {
    "network": plot_network,
    "node_metrics": plot_node_metrics,
}

# Signed-laplacian is per-filter but only for filters with negative edges.
_PLOTS_SIGNED_ONLY: dict[str, Callable] = {
    "signed_laplacian": plot_signed_laplacian_spectrum,
}

# LRG plots: one figure per filter, only when lrg_results exist.
_PLOTS_LRG: dict[str, Callable] = {
    "lrg_entropy": plot_lrg_entropy,
    "lrg_dendrogram": plot_lrg_dendrogram,
    "lrg_psi": plot_lrg_psi,
    "lrg_partition": plot_lrg_partition_network,
    "lrg_sankey": plot_lrg_sankey,
}

_ALL_PLOT_NAMES = (
    set(_PLOTS_PER_RESULT) | set(_PLOTS_PER_FILTER)
    | set(_PLOTS_SIGNED_ONLY) | set(_PLOTS_LRG)
)


_PLOT_DESCRIPTIONS: dict[str, str] = {
    "summary": (
        "Six-panel overview of the pipeline result.  Top row: correlation "
        "matrix heatmap, weight distribution (positive/negative histogram), "
        "eigenvalue spectrum.  Bottom row: signed Laplacian spectrum, "
        "filtered-network density comparison, node-strength distribution "
        "for the first filter."
    ),
    "correlation_matrix": (
        "Heatmap of the cleaned correlation matrix using a diverging "
        "blue-white-red colourmap.  Values range from -1 (anti-correlated) "
        "to +1 (correlated).  Dead brain regions have been removed."
    ),
    "weight_distribution": (
        "Histogram of pairwise correlation weights split into positive "
        "(blue) and negative (red) contributions.  The vertical dashed "
        "line marks zero.  Counts in the legend."
    ),
    "signed_balance": (
        "Signed-network balance profile.  Left: scatter of per-node "
        "positive strength vs negative strength — points on the diagonal "
        "have equal positive and negative connections.  Right: histogram "
        "of per-node balance ratio s⁺/(s⁺+s⁻); values near 1 indicate "
        "mostly positive connections, near 0.5 indicates balanced."
    ),
    "signed_network": (
        "Full signed correlation network drawn with spring-layout.  Blue "
        "edges = positive correlations, red edges = negative.  Edge width "
        "and opacity scale with |correlation|."
    ),
    "filtered_comparison": (
        "Three-panel comparison of the same data under different filters.  "
        "Left: absolute-value network (thresholded at the percolation "
        "threshold Th).  Centre: positive-only network.  Right: full "
        "signed network laid out with signed-Laplacian spectral embedding "
        "(positive edges blue, negative edges red).  Edge thickness and "
        "transparency scale with |weight|."
    ),
    "signed_laplacian": (
        "Eigenvalue spectrum of the signed Laplacian L=|D|-A.  Negative "
        "eigenvalues (red bars) indicate structural imbalance / frustration "
        "in the signed network.  Title shows the frustration index."
    ),
    "network": (
        "Force-directed layout of the filtered and thresholded network.  "
        "Nodes coloured by Louvain community; node size proportional to "
        "strength.  Edge width scales with weight."
    ),
    "node_metrics": (
        "Distribution histograms of per-node metrics (degree, strength, "
        "clustering coefficient, betweenness centrality) for one filtered "
        "network.  Red dashed line marks the mean."
    ),
    "lrg_entropy": (
        "LRG diffusion entropy (1-S, blue) and specific heat (C, red) on "
        "dual y-axes as a function of diffusion time t.  Peaks in C "
        "correspond to characteristic scales of the network."
    ),
    "lrg_dendrogram": (
        "Hierarchical clustering dendrogram from the LRG analysis at a "
        "specific tau.  The red dashed line shows the flat-clustering "
        "threshold used to assign communities."
    ),
    "lrg_psi": (
        "Partition Stability Index (PSI) bar chart.  Each bar is the "
        "stability of a dendrogram branch.  The red dashed line marks "
        "the optimal branch (maximum stability)."
    ),
    "lrg_partition": (
        "Filtered network with nodes coloured by the LRG partition "
        "(not Louvain).  Layout identical to the network plot.  Title "
        "shows the diffusion-time tau and the number of clusters."
    ),
    "lrg_sankey": (
        "Sankey (alluvial) diagram showing how communities evolve across "
        "diffusion timescales tau.  Each vertical column is a partition "
        "at a given tau; flows connect matching nodes between scales."
    ),
}


def _save_plot(fig, out_dir: Path, label: str, name: str,
               fmt: str, dpi: int, plt_mod) -> Path:
    """Save a figure + .txt description preserving label directory structure."""
    rel = Path(label or "unlabeled")
    subdir = out_dir / rel.parent
    subdir.mkdir(parents=True, exist_ok=True)
    fpath = subdir / f"{rel.name}_{name}.{fmt}"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt_mod.close(fig)

    # Write description alongside the figure.
    # Strip filter prefix (e.g. "absolute_network" → "network").
    base_name = name
    for prefix in ("absolute_", "positive_", "negative_"):
        if name.startswith(prefix):
            base_name = name[len(prefix):]
            break
    desc = _PLOT_DESCRIPTIONS.get(base_name, f"Plot: {name}")
    txt_path = fpath.with_suffix(".txt")
    txt_path.write_text(desc + "\n")

    return fpath


def _cmd_plot(args: argparse.Namespace) -> int:
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    input_path: Path = args.input.resolve()
    results = load_results(input_path)

    if args.pattern:
        results = results.filter(args.pattern)

    if len(results) == 0:
        print("No results match the given pattern.", file=sys.stderr)
        return 1

    # Determine which plots to produce.
    if args.plots:
        requested = set(args.plots)
    else:
        requested = set(_ALL_PLOT_NAMES)

    unknown = requested - _ALL_PLOT_NAMES
    if unknown:
        print(
            f"Unknown plot name(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(_ALL_PLOT_NAMES))}",
            file=sys.stderr,
        )
        return 1

    req_per_result = requested & set(_PLOTS_PER_RESULT)
    req_per_filter = requested & set(_PLOTS_PER_FILTER)
    req_signed_only = requested & set(_PLOTS_SIGNED_ONLY)
    req_lrg = requested & set(_PLOTS_LRG)

    # Output directory: default to <results_dir>/figures/
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif input_path.is_dir():
        output_dir = input_path / "figures"
    else:
        output_dir = input_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.fmt
    dpi = args.dpi
    n_saved = 0

    pbar = tqdm(list(results), desc="Plotting", unit="matrix",
                file=sys.stderr, dynamic_ncols=True)
    for r in pbar:
        label = r.label or "unlabeled"
        pbar.set_postfix_str(label, refresh=True)

        # ── Per-result plots ─────────────────────────────────────────
        for name in sorted(req_per_result):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig, _ = _PLOTS_PER_RESULT[name](r)
                _save_plot(fig, output_dir, label, name, fmt, dpi, plt)
                n_saved += 1
            except Exception:
                pass  # skip gracefully

        # ── Per-filter plots ─────────────────────────────────────────
        if r.network_analyses and (req_per_filter or req_signed_only or req_lrg):
            for fname in r.filtered_networks:
                for name in sorted(req_per_filter):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fn = _PLOTS_PER_FILTER[name]
                            fig, _ = fn(r, filter_name=fname)
                        _save_plot(fig, output_dir, label,
                                   f"{fname}_{name}", fmt, dpi, plt)
                        n_saved += 1
                    except Exception:
                        pass

                # ── Signed-laplacian: only for the "negative" filter
                #    (or any filter known to have negative edges).
                if fname == "negative" and req_signed_only:
                    for name in sorted(req_signed_only):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                fig, _ = plot_signed_laplacian_spectrum(r)
                            _save_plot(fig, output_dir, label,
                                       f"{fname}_{name}", fmt, dpi, plt)
                            n_saved += 1
                        except Exception:
                            pass

                # ── LRG plots (only if lrg_results exist for this filter)
                if r.lrg_results and r.lrg_results.get(fname):
                    lrg_parts = r.lrg_results[fname]
                    # Skip if it's an error entry.
                    if isinstance(lrg_parts, list) and len(lrg_parts) > 0:
                        for name in sorted(req_lrg):
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    fn = _PLOTS_LRG[name]
                                    fig, _ = fn(r, filter_name=fname)
                                _save_plot(fig, output_dir, label,
                                           f"{fname}_{name}", fmt, dpi, plt)
                                n_saved += 1
                            except Exception:
                                pass

    pbar.close()
    print(
        f"Saved {n_saved} plots for {len(results)} matrices to {output_dir}/",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the ``multifunbrain`` console script."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    func: CommandHandler = args.func
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
