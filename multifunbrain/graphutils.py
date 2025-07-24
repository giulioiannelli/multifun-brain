from .core import *

from networkx import Graph

def get_giant_component(G, strongly=False):
    """
    Returns the subgraph corresponding to the giant component of a graph.

    Parameters:
        G (networkx.Graph or DiGraph): The input graph.
        strongly (bool): If G is directed, whether to use strongly connected components
                         (True) or weakly connected components (False).

    Returns:
        networkx.Graph or DiGraph: The subgraph of the largest component.
    """
    if G.is_directed():
        if strongly:
            components = nx.strongly_connected_components(G)
        else:
            components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    largest_component = max(components, key=len)
    return G.subgraph(largest_component).copy()

def get_giant_component_leftoff(graph: nx.Graph, remove_selfloops: bool = True) -> Tuple[nx.Graph, List]:
    """
    Remove self-loops (optional) from the input graph and return its largest connected component,
    along with the list of nodes that are not part of the giant component.

    Parameters
    ----------
    graph : nx.Graph
        The input graph from which self-loops are optionally removed.
    remove_selfloops : bool, optional
        Whether to remove self-loops from the graph (default is True).

    Returns
    -------
    Tuple[nx.Graph, List]
        A tuple containing:
          - A copy of the largest connected component of the cleaned graph.
          - A list of node names that are not included in the giant component.
    """
    G_clean = graph.copy()
    if remove_selfloops:
        G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
    
    components = sorted(nx.connected_components(G_clean), key=len, reverse=True)
    if components:
        giant_nodes = components[0]
    else:
        giant_nodes = set()
    
    giant_component = G_clean.subgraph(giant_nodes).copy()
    left_off_nodes = list(set(G_clean.nodes()) - giant_nodes)
    
    return giant_component, left_off_nodes

def build_correlation_network(ts, regularize=True, remove_negative=True, blank_diagonal=True, threshold=None):
    """
    Build a correlation network from time series, with optional thresholding.

    Parameters:
    -----------
    ts : ndarray (n_signals x n_timepoints)
        Input time series matrix.

    regularize : bool, default=True
        If True, replace NaNs in the correlation matrix with 0.

    remove_negative : bool, default=True
        If True, set negative correlations to 0.

    blank_diagonal : bool, default=True
        If True, set diagonal elements to 0 (remove self-loops).

    threshold : float or None, optional
        If given, sets to zero all correlation values strictly below this threshold.

    Returns:
    --------
    G : networkx.Graph
        The graph built from the processed correlation matrix, restricted to its largest component.

    remnodes : list of int
        List of node indices removed (not in the giant component).
    """
    C = np.corrcoef(ts)

    if remove_negative:
        C[C < 0] = 0

    if regularize:
        C = np.nan_to_num(C, nan=0.0)

    if threshold is not None:
        C[C < threshold] = 0

    if blank_diagonal:
        np.fill_diagonal(C, 0)

    G = nx.from_numpy_array(C)
    G, remnodes = get_giant_component_leftoff(G)

    return G, remnodes

def compute_threshold_stats(G0: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute threshold values and connectivity statistics for a weighted graph.

    The function generates a logarithmically spaced array of threshold values between 1e-5 and the 
    logarithm of the maximum edge weight. For each threshold, it removes edges below the threshold 
    and calculates:
      - Pinf: the fraction of nodes in the largest connected component relative to G0.
      - Einf: the fraction of edges in the largest connected component relative to G0.
      
    Parameters
    ----------
    G0 : nx.Graph
        A weighted graph (assumed to be the largest connected component) where each edge 
        has a "weight" attribute.

    Returns
    -------
    Th : np.ndarray
        Array of threshold values.
    Einf : np.ndarray
        Array of fractions of edges in the giant component at each threshold.
    Pinf : np.ndarray
        Array of fractions of nodes in the giant component at each threshold.
    """
    weights = [data["weight"] for _, _, data in G0.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)
    Th = np.logspace(np.log10(min_weight), np.log10(max_weight), 400)
    Pinf = np.zeros(len(Th))
    Einf = np.zeros(len(Th))
    E0 = len(G0.edges())
    
    for i, threshold in enumerate(Th):
        F = G0.copy()
        F.remove_edges_from([(u, v) for u, v, w in F.edges(data="weight") if w < threshold])
        if F.number_of_edges() == 0:
            Pinf[i] = 0
            Einf[i] = 0
        else:
            components = sorted(nx.connected_components(F), key=len, reverse=True)
            giant = F.subgraph(components[0]).copy()
            Pinf[i] = len(giant.nodes()) / len(G0.nodes())
            Einf[i] = len(giant.edges()) / E0
    return Th, Einf, Pinf

def compute_threshold_stats_fast(G0: Graph, n_points: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast computation of threshold values and connectivity statistics using Union-Find.
    
    This is an optimized version that uses Union-Find data structure for efficient
    connected component computation, especially beneficial for large graphs.

    Parameters
    ----------
    G0 : Graph
        A weighted graph (assumed to be the largest connected component) where each edge 
        has a "weight" attribute.
    n_points : int, optional
        Number of threshold points to compute. If 0, uses the number of nodes in G0.

    Returns
    -------
    Th : np.ndarray
        Array of threshold values.
    Einf : np.ndarray
        Array of fractions of edges in the giant component at each threshold.
    Pinf : np.ndarray
        Array of fractions of nodes in the giant component at each threshold.
    """
    # Pre-compute all edge data
    edges_data = [(u, v, data["weight"]) for u, v, data in G0.edges(data=True)]
    weights = np.array([w for _, _, w in edges_data])
    
    # Create node mapping for Union-Find
    nodes = list(G0.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    n_points = n_points or n_nodes
    Th = np.logspace(np.log10(weights.min()), np.log10(weights.max()), n_points)
    Pinf = np.zeros(len(Th))
    Einf = np.zeros(len(Th))
    
    E0 = len(edges_data)
    
    # Sort edges by weight (descending) for efficient processing
    edge_indices = np.argsort(weights)[::-1]
    sorted_edges = [(edges_data[i][0], edges_data[i][1], weights[i]) for i in edge_indices]
    
    for i, threshold in enumerate(Th):
        # Union-Find implementation
        parent = list(range(n_nodes))
        rank = [0] * n_nodes
        component_size = [1] * n_nodes
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            component_size[px] += component_size[py]
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Add edges above threshold
        valid_edge_count = 0
        for u, v, w in sorted_edges:
            if w >= threshold:
                union(node_to_idx[u], node_to_idx[v])
                valid_edge_count += 1
            else:
                break  # Since edges are sorted by weight, we can break early
        
        if valid_edge_count == 0:
            Pinf[i] = 0
            Einf[i] = 0
        else:
            # Find the largest component
            max_component_size = max(component_size[find(j)] for j in range(n_nodes))
            
            # Count edges in the largest component
            giant_root = None
            for j in range(n_nodes):
                if component_size[find(j)] == max_component_size:
                    giant_root = find(j)
                    break
            
            giant_nodes = {nodes[j] for j in range(n_nodes) if find(j) == giant_root}
            giant_edges = sum(1 for u, v, w in edges_data 
                            if w >= threshold and u in giant_nodes and v in giant_nodes)
            
            Pinf[i] = max_component_size / n_nodes
            Einf[i] = giant_edges / E0
    
    return Th, Einf, Pinf

def select_threshold_elbow(Th: np.ndarray, Pinf: np.ndarray) -> float:
    """
    Select threshold at the point of steepest decline in Pinf (elbow method).

    Parameters
    ----------
    Th : np.ndarray
        Array of threshold values.
    Pinf : np.ndarray
        Fraction of nodes in the giant component at each threshold.

    Returns
    -------
    float
        Threshold value corresponding to the steepest drop in Pinf.
    """
    dP = -np.gradient(Pinf, Th)           # approximate first derivative
    idx = np.argmax(dP)                   # max drop
    return Th[idx]


def select_threshold_fraction(Th: np.ndarray, Pinf: np.ndarray, drop_frac: float = 0.95) -> float:
    """
    Select the first threshold where Pinf drops below a given fraction of its initial value.

    Parameters
    ----------
    Th : np.ndarray
        Array of threshold values.
    Pinf : np.ndarray
        Fraction of nodes in the giant component at each threshold.
    drop_frac : float, optional
        Fractional threshold of initial Pinf value (default is 0.95).

    Returns
    -------
    float
        Threshold value where Pinf first drops below `drop_frac * P0`.
    """
    P0 = Pinf[0]
    for i in range(len(Pinf)):
        if Pinf[i] < drop_frac * P0:
            return Th[i]
    return Th[-1]

def select_threshold_plateau(Th: np.ndarray, Pinf: np.ndarray, tol: float = 0.01) -> float:
    """
    Select the largest threshold such that Pinf remains within `tol` of its maximum.

    Parameters
    ----------
    Th : np.ndarray
        Array of threshold values.
    Pinf : np.ndarray
        Fraction of nodes in the giant component at each threshold.
    tol : float, optional
        Allowed drop from the maximum Pinf (default is 0.01).

    Returns
    -------
    float
        Highest threshold where Pinf is still within `tol` of its maximum.
    """
    max_val = np.max(Pinf)
    for i in reversed(range(len(Pinf))):
        if Pinf[i] >= max_val - tol:
            return Th[i]
    return Th[0]

def compute_normalized_linkage(dists, G, method="average", labelList: str = "names"):
    # Initial linkage computation.
    linkage_matrix = linkage(dists, method)
    # Create labels for nodes.
    if labelList == "names":
        label_list = list(G.nodes())
    elif labelList == "numbers":
        label_list = [i + 1 for i in range(len(G.nodes()))]
    # Determine tmax as the maximum merge distance plus 1% of that distance.
    max_distance = linkage_matrix[-1, 2]
    tmax = max_distance + 0.01 * max_distance
    # Recompute the linkage matrix using normalized distances.
    linkage_matrix = linkage(dists / tmax, method)
    return linkage_matrix, label_list, tmax

def compute_optimal_threshold_std(merge_distances):
    diffs = np.diff(merge_distances)
    print("diffs", diffs)
    
    max_gap_index = np.argmax(diffs)
    print("max_gap_index", max_gap_index)
    
    optimal_threshold = (merge_distances[max_gap_index] + merge_distances[max_gap_index + 1]) / 2
    return optimal_threshold


def find_threshold_jumps(G_filt):
    Th, Einf, Pinf = compute_threshold_stats_fast(G_filt)
    Pinf_diff = np.diff(Pinf)
    jumps = np.where(Pinf_diff != 0)[0]
    return Th, jumps