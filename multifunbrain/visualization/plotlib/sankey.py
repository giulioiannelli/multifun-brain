"""Sankey diagrams of partition flow across τ scales.

Both matplotlib and Plotly backends live here behind a single entry
point: ``plot_sankey(partitions, tau_values, backend=...)``. The
``plot_sankey_matplotlib`` alias is kept as a thin wrapper for callers
that imported it under the old layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from . import Polygon, Rectangle, go, plt

__all__ = [
    "plot_sankey",
    "plot_sankey_matplotlib",
]


def plot_sankey(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
    *,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
) -> None:
    """Sankey diagram of community evolution across τ stages.

    Parameters
    ----------
    partitions:
        ``partitions[k][i]`` is the cluster label of node ``i`` at the
        ``k``-th τ stage.
    tau_values:
        τ value labelling each stage.
    backend:
        ``"plotly"`` (default) renders an interactive Sankey via
        :class:`plotly.graph_objects.Sankey`. ``"matplotlib"`` falls back
        to a hand-drawn rectangle/polygon Sankey suitable for static PDFs.
    **kwargs:
        Forwarded to the chosen backend (currently only the matplotlib
        backend honours ``total_height``, ``col_gap``, ``block_width``).
    """
    if backend == "plotly":
        return _plot_sankey_plotly(partitions, tau_values)
    if backend == "matplotlib":
        return _plot_sankey_matplotlib(partitions, tau_values, **kwargs)
    raise ValueError(
        f"Unknown sankey backend {backend!r} — expected 'plotly' or 'matplotlib'."
    )


def plot_sankey_matplotlib(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
    total_height: float = 10,
    col_gap: float = 1,
    block_width: float = 0.1,
) -> None:
    """Back-compat alias — equivalent to ``plot_sankey(..., backend='matplotlib')``."""
    return _plot_sankey_matplotlib(
        partitions, tau_values,
        total_height=total_height, col_gap=col_gap, block_width=block_width,
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _plot_sankey_plotly(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
) -> None:
    n_stages = len(partitions)
    sankey_labels: list[str] = []
    stage_mapping: list[dict[int, int]] = []
    idx = 0
    for i in range(n_stages):
        clusters = sorted(set(partitions[i]))
        mapping: dict[int, int] = {}
        for c in clusters:
            label = f"τ={tau_values[i]}: Cluster {c}"
            sankey_labels.append(label)
            mapping[c] = idx
            idx += 1
        stage_mapping.append(mapping)

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    n_nodes = len(partitions[0])
    for stage in range(n_stages - 1):
        flow: dict[tuple[int, int], int] = {}
        for node in range(n_nodes):
            src = stage_mapping[stage][partitions[stage][node]]
            trg = stage_mapping[stage + 1][partitions[stage + 1][node]]
            flow[(src, trg)] = flow.get((src, trg), 0) + 1
        for (src, trg), val in flow.items():
            sources.append(src)
            targets.append(trg)
            values.append(val)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_labels,
            ),
            link=dict(source=sources, target=targets, value=values),
        )
    )

    fig.update_layout(
        title_text="Sankey Diagram of Metastable Node Transitions",
        font_size=10,
    )
    fig.show()


def _plot_sankey_matplotlib(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
    total_height: float = 10,
    col_gap: float = 1,
    block_width: float = 0.1,
) -> None:
    n_stages = len(partitions)
    n_nodes = len(partitions[0])
    h_unit = total_height / n_nodes

    stage_positions = []
    for stage in range(n_stages):
        clusters: dict[int, int] = {}
        for c in partitions[stage]:
            clusters[c] = clusters.get(c, 0) + 1
        sorted_clusters = sorted(clusters.keys())
        pos: dict[int, dict[str, float]] = {}
        current_y = total_height
        for cl in sorted_clusters:
            count = clusters[cl]
            height = count * h_unit
            pos[cl] = {
                "top": current_y,
                "bottom": current_y - height,
                "pointer_out": current_y,
                "pointer_in": current_y,
            }
            current_y -= height
        stage_positions.append(pos)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_stages):
        x = i * col_gap
        for cl, pos in stage_positions[i].items():
            rect = Rectangle(
                (x - block_width, pos["bottom"]),
                2 * block_width,
                pos["top"] - pos["bottom"],
                facecolor="lightgray",
                edgecolor="black",
                lw=1,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                (pos["top"] + pos["bottom"]) / 2,
                f"{tau_values[i]}\ncl {cl}",
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )

    flows: list[dict[str, float]] = []
    for stage in range(n_stages - 1):
        x_left = stage * col_gap
        x_right = (stage + 1) * col_gap
        for node in range(n_nodes):
            c_left = partitions[stage][node]
            c_right = partitions[stage + 1][node]
            left_pos = stage_positions[stage][c_left]
            y_left_start = left_pos["pointer_out"]
            y_left_end = y_left_start - h_unit
            left_pos["pointer_out"] = y_left_end

            right_pos = stage_positions[stage + 1][c_right]
            y_right_start = right_pos["pointer_in"]
            y_right_end = y_right_start - h_unit
            right_pos["pointer_in"] = y_right_end

            flows.append(
                {
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_left_top": y_left_start,
                    "y_left_bot": y_left_end,
                    "y_right_top": y_right_start,
                    "y_right_bot": y_right_end,
                }
            )

    for f in flows:
        verts = [
            (f["x_left"] + block_width, f["y_left_top"]),
            (f["x_left"] + block_width, f["y_left_bot"]),
            (f["x_right"] - block_width, f["y_right_bot"]),
            (f["x_right"] - block_width, f["y_right_top"]),
        ]
        poly = Polygon(
            verts,
            closed=True,
            facecolor="skyblue",
            edgecolor="none",
            alpha=0.5,
            zorder=1,
        )
        ax.add_patch(poly)

    ax.set_xlim(-col_gap, n_stages * col_gap)
    ax.set_ylim(0, total_height)
    ax.axis("off")
    plt.title("Sankey Diagram of Metastable Node Transitions")
    plt.show()
