import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import plotly.graph_objects as go


def plot_entropy_and_C(ax1, t, Sm1, Csp, color1="blue", color2="red"):
    """
    Plot 1-S and C on the same x-axis (log scale) with twin y-axes.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Primary axis to plot 1-S.

    t : array-like
        Time values (must match Sm1 and Csp).

    Sm1 : array-like
        Entropy-related quantity to plot as 1-S.

    Csp : array-like
        Clustering coefficient or similar metric (must be len(t) - 1).

    color1 : str, optional
        Color for 1-S plot (default is "blue").

    color2 : str, optional
        Color for C plot (default is "red").
    """
    ax1.plot(t, Sm1, label=r"$1-S$", color=color1)
    ax1.set_ylabel("1-S", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.plot(t[:-1], Csp, label=r"$C$", color=color2)
    ax2.set_ylabel("C", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_xlabel(r"$\tau$")


def plot_sankey_matplotlib(partitions, tau_values, total_height=10, col_gap=1, block_width=0.1):
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
        clusters = {}
        # Count nodes per cluster
        for c in partitions[stage]:
            clusters[c] = clusters.get(c, 0) + 1
        # Sort clusters to get a fixed order (can be changed)
        sorted_clusters = sorted(clusters.keys())
        pos = {}
        current_y = total_height
        for cl in sorted_clusters:
            count = clusters[cl]
            height = count * h_unit
            pos[cl] = {'top': current_y, 'bottom': current_y - height,
                       'pointer_out': current_y, 'pointer_in': current_y}  # start pointer at top
            current_y -= height
        stage_positions.append(pos)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw the blocks for each stage as rectangles.
    for i in range(n_stages):
        x = i * col_gap
        for cl, pos in stage_positions[i].items():
            rect = Rectangle((x - block_width, pos['bottom']),
                             2*block_width, pos['top'] - pos['bottom'],
                             facecolor='lightgray', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(rect)
            # Label block with cluster ID and τ value
            ax.text(x, (pos['top'] + pos['bottom'])/2, f"{tau_values[i]}\ncl {cl}",
                    ha='center', va='center', fontsize=8, zorder=3)

    # For flows between consecutive stages, allocate vertical segments.
    flows = []  # each element: dict with stage, from cluster, to cluster, and segment coordinates.
    for stage in range(n_stages - 1):
        x_left = stage * col_gap
        x_right = (stage + 1) * col_gap
        # Iterate over nodes in order (by index)
        for node in range(n_nodes):
            c_left = partitions[stage][node]
            c_right = partitions[stage+1][node]
            # Allocate outgoing segment from left block:
            left_pos = stage_positions[stage][c_left]
            y_left_start = left_pos['pointer_out']
            y_left_end = y_left_start - h_unit
            left_pos['pointer_out'] = y_left_end  # update pointer
            
            # Allocate incoming segment in right block:
            right_pos = stage_positions[stage+1][c_right]
            y_right_start = right_pos['pointer_in']
            y_right_end = y_right_start - h_unit
            right_pos['pointer_in'] = y_right_end  # update pointer
            
            flows.append({
                'x_left': x_left,
                'x_right': x_right,
                'y_left_top': y_left_start,
                'y_left_bot': y_left_end,
                'y_right_top': y_right_start,
                'y_right_bot': y_right_end
            })
    
    # Draw flow polygons between stages.
    for f in flows:
        # Define polygon vertices:
        verts = [
            (f['x_left'] + block_width, f['y_left_top']),
            (f['x_left'] + block_width, f['y_left_bot']),
            (f['x_right'] - block_width, f['y_right_bot']),
            (f['x_right'] - block_width, f['y_right_top']),
        ]
        poly = Polygon(verts, closed=True, facecolor='skyblue', edgecolor='none', alpha=0.5, zorder=1)
        ax.add_patch(poly)

    # Set limits and remove axes for clarity.
    ax.set_xlim(-col_gap, n_stages * col_gap)
    ax.set_ylim(0, total_height)
    ax.axis('off')
    plt.title("Sankey Diagram of Metastable Node Transitions")
    plt.show()


def plot_sankey(partitions, tau_values):
    """
    partitions: list of lists containing community labels for each node at each τ.
    tau_values: list of τ values corresponding to each partition.
    """
    n_stages = len(partitions)
    sankey_labels = []
    stage_mapping = []  # mapping of each stage's cluster to sankey node index
    idx = 0
    for i in range(n_stages):
        clusters = sorted(set(partitions[i]))
        mapping = {}
        for c in clusters:
            label = f"τ={tau_values[i]}: Cluster {c}"
            sankey_labels.append(label)
            mapping[c] = idx
            idx += 1
        stage_mapping.append(mapping)

    sources, targets, values = [], [], []
    n_nodes = len(partitions[0])
    for stage in range(n_stages - 1):
        flow = {}
        for node in range(n_nodes):
            src = stage_mapping[stage][partitions[stage][node]]
            trg = stage_mapping[stage+1][partitions[stage+1][node]]
            flow[(src, trg)] = flow.get((src, trg), 0) + 1
        for (src, trg), val in flow.items():
            sources.append(src)
            targets.append(trg)
            values.append(val)

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))
    
    fig.update_layout(title_text="Sankey Diagram of Metastable Node Transitions", font_size=10)
    fig.show()