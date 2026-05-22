"""Matplotlib-based Sankey visualisations."""

from __future__ import annotations

from collections.abc import Sequence

from . import Polygon, Rectangle, plt


def plot_sankey_matplotlib(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
    total_height: float = 10,
    col_gap: float = 1,
    block_width: float = 0.1,
) -> None:
    """
    partitions: list of lists/arrays of community labels (one per node)
                for each stage (τ value).
    tau_values: list of τ values corresponding to each partition.
    total_height: vertical span for the diagram.
    col_gap: horizontal gap between stages.
    block_width: half-width of node blocks from the stage x coordinate.
    """
    n_stages = len(partitions)
    n_nodes = len(partitions[0])
    h_unit = total_height / n_nodes  # vertical size per node

    # Compute the block positions per stage.
    # For each stage, for each cluster, assign a rectangle spanning the aggregated flow.
    stage_positions = []  # list of dict: stage_positions[i][cluster] = dict with keys 'top', 'bottom',
    # and pointers for outgoing and incoming flow allocation.
    for stage in range(n_stages):
        clusters: dict[int, int] = {}
        # Count nodes per cluster
        for c in partitions[stage]:
            clusters[c] = clusters.get(c, 0) + 1
        # Sort clusters to get a fixed order (can be changed)
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
            }  # start pointer at top
            current_y -= height
        stage_positions.append(pos)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the blocks for each stage as rectangles.
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
            # Label block with cluster ID and τ value
            ax.text(
                x,
                (pos["top"] + pos["bottom"]) / 2,
                f"{tau_values[i]}\ncl {cl}",
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )

    # For flows between consecutive stages, allocate vertical segments.
    flows: list[dict[str, float]] = []
    for stage in range(n_stages - 1):
        x_left = stage * col_gap
        x_right = (stage + 1) * col_gap
        # Iterate over nodes in order (by index)
        for node in range(n_nodes):
            c_left = partitions[stage][node]
            c_right = partitions[stage + 1][node]
            # Allocate outgoing segment from left block:
            left_pos = stage_positions[stage][c_left]
            y_left_start = left_pos["pointer_out"]
            y_left_end = y_left_start - h_unit
            left_pos["pointer_out"] = y_left_end  # update pointer

            # Allocate incoming segment in right block:
            right_pos = stage_positions[stage + 1][c_right]
            y_right_start = right_pos["pointer_in"]
            y_right_end = y_right_start - h_unit
            right_pos["pointer_in"] = y_right_end  # update pointer

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

    # Draw flow polygons between stages.
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

    # Set limits and remove axes for clarity.
    ax.set_xlim(-col_gap, n_stages * col_gap)
    ax.set_ylim(0, total_height)
    ax.axis("off")
    plt.title("Sankey Diagram of Metastable Node Transitions")
    plt.show()
