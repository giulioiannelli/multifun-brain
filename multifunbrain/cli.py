"""Command-line interface for :mod:`multifunbrain`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import networkx as nx

from .core import hello_brain
from .generation import generate_hmn

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

    return parser


def _cmd_hello(args: argparse.Namespace) -> int:
    print(hello_brain(args.name))
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

    stats: Dict[str, Any] = {
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``multifunbrain`` console script."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    func: CommandHandler = getattr(args, "func")
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
