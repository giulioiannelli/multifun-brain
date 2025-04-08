from .core import *



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