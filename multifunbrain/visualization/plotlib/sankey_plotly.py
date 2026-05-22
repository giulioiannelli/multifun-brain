"""Plotly-based Sankey visualisations."""

from __future__ import annotations

from collections.abc import Sequence

from . import go


def plot_sankey(
    partitions: Sequence[Sequence[int]],
    tau_values: Sequence,
) -> None:
    """
    partitions: list of lists containing community labels for each node at each τ.
    tau_values: list of τ values corresponding to each partition.
    """
    n_stages = len(partitions)
    sankey_labels: list[str] = []
    stage_mapping: list[dict[int, int]] = []  # mapping of each stage's cluster to sankey node index
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
