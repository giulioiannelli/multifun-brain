from .core import *

def rho_matrix(tau, L):
    """
    Compute the normalized diffusion kernel (a stochastic matrix) 
    based on the Laplacian `L` at time scale `tau`.

    Parameters:
    -----------
    tau : float
        Diffusion time scale.

    L : ndarray (n x n)
        Graph Laplacian matrix (assumed symmetric and positive semi-definite).

    Returns:
    --------
    rho : ndarray (n x n)
        Normalized diffusion matrix with trace 1.
    """
    kernel = expm(-tau * L)               # diffusion kernel
    trace = np.trace(kernel)              # normalization constant
    return kernel / trace                 # normalized matrix

def entropy(w, steps=600, t1=-2, t2=5):
    """
    Compute the diffusion-based spectral entropy of a graph.

    Parameters:
        L (ndarray): Laplacian matrix (unused here but kept for interface consistency).
        w (ndarray): Spectrum (eigenvalues of L).
        steps (int): Number of diffusion time steps.
        wTresh (float): Threshold to ignore small eigenvalues.
        t1 (float): log10(start time)
        t2 (float): log10(end time)

    Returns:
        S (ndarray): Normalized entropy as a function of time.
        dS (ndarray): Derivative of 1 - S (rate of entropy change).
        VarL (ndarray): Laplacian spectral variance over time.
        t (ndarray): Time points (logspaced).
    """
    N = len(w)
    t = np.logspace(t1, t2, int(steps))
    S = np.zeros(len(t))
    VarL = np.zeros(len(t))

    for i, tau in enumerate(t):
        rhoTr = np.exp(-tau * w)
        Tr = np.sum(rhoTr)
        rho = rhoTr / Tr

        S[i] = -np.nansum(rho * np.log(rho)) / np.log(N)

        avg = np.sum(w * rhoTr) / Tr
        avg2 = np.sum((w ** 2) * rhoTr) / Tr
        VarL[i] = avg2 - avg**2

    dS = np.log(N) * np.diff(1 - S) / np.diff(np.log(t))
    return 1 - S, dS, VarL, t


def symmetrized_inverse_distance(tau, rho_matrix_fn):
    """
    Compute a symmetric distance matrix from an input correlation-like matrix.

    Parameters:
    -----------
    tau : any
        Input parameter passed to the `rho_matrix_fn`, typically a time or scale.
    
    rho_matrix_fn : callable
        A function that returns a square matrix (e.g., correlation matrix) given `tau`.

    Returns:
    --------
    dists : ndarray
        Condensed 1D array of the symmetric distance matrix, suitable for clustering methods.
    """
    Trho = 1.0 / rho_matrix_fn(tau)
    Trho = np.maximum(Trho, Trho.T)     # ensure symmetry
    np.fill_diagonal(Trho, 0)           # zero diagonal (no self-distance)
    return squareform(Trho)             # convert to condensed form

def compute_optimal_threshold(linkage_matrix, scaling_factor=1):
    """
    Compute the optimal flat clustering threshold from a linkage matrix
    using the partition stability index.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from hierarchical clustering. It is assumed that 
        the third column contains the merge distances.
    scaling_factor : float, optional
        A factor to scale the optimal threshold (default is 0.9).

    Returns
    -------
    FlatClusteringTh : float
        The computed threshold for flat clustering.
    optimal_threshold : float
        The optimal threshold derived from the dendrogram gap analysis.
    stability_indices : np.ndarray
        The array of computed stability indices for each branch.
    optimal_branch_index : int
        The index corresponding to the branch with the maximum stability index.
    """
    # Extract merge distances from the linkage matrix
    dendro_thresholds = linkage_matrix[:, 2]
    # Reverse order so that the first element corresponds to the initial split
    D_values = dendro_thresholds[::-1]
    
    # Compute the normalization constant N using the first and last thresholds
    N = 1 / (np.log10(D_values[0]) - np.log10(D_values[-1]))
    
    # Compute the stability index for each dendrogram gap
    stability_indices = []
    for i in range(len(D_values) - 1):
        sigma = N * (np.log10(D_values[i]) - np.log10(D_values[i+1]))
        stability_indices.append(sigma)
    stability_indices = np.array(stability_indices)
    
    # Identify the branch with the highest stability index
    optimal_branch_index = np.argmax(stability_indices)
    # The optimal threshold is D_(n+1) corresponding to that branch
    optimal_threshold = D_values[optimal_branch_index + 1]
    
    # Apply a scaling factor to determine the final flat clustering threshold
    FlatClusteringTh = optimal_threshold * scaling_factor

    return FlatClusteringTh, optimal_threshold, stability_indices, optimal_branch_index

def identify_switching_nodes(partitions, tau_values):
    """
    partitions: list of lists of community labels for each node at each τ scale.
    tau_values: list of τ values corresponding to each partition.
    
    Returns a dictionary where each key is a node index (int) that switched communities,
    and the value is a list of (τ, community) tuples showing its assignment history.
    Only nodes with more than one unique community label are included.
    """
    if not partitions:
        return {}
    
    n_nodes = len(partitions[0])
    result = {}
    for node in range(n_nodes):
        history = [(float(tau_values[i]), int(partitions[i][node])) for i in range(len(tau_values))]
        # Include node if it is assigned to different communities across τ scales.
        if len({assignment for _, assignment in history}) > 1:
            result[node] = history
    return result

def get_moved_nodes(partdict_tau, source_cluster):
    """
    partdict_tau: dictionary in which keys are node IDs and values are lists of (τ, cluster) tuples,
                  e.g., {0: [(0.01, 3), (0.07485, 3), (1.0, 2)], ...}
    source_cluster: the cluster of interest (the initial cluster in which nodes must be)
    
    Returns a dictionary mapping node IDs that start in source_cluster and later move out,
    with details of the first τ at which they change and what the new cluster is.
    """
    moved = {}
    for node, history in partdict_tau.items():
        # Only consider nodes that start in the specified source cluster.
        if history[0][1] == source_cluster:
            # Look for the first occurrence where the cluster assignment differs.
            for tau, cluster in history:
                if cluster != source_cluster:
                    moved[node] = {"tau_move": tau, "new_cluster": cluster, "history": history}
                    break  # record the first move only
    return moved

def get_moved_nodes_interval(partdict_tau, source_cluster, tau_i=None, tau_f=None, tol=1e-3):
    """
    partdict_tau: dict mapping node IDs to lists of (τ, cluster) tuples.
                  Example:
                    {0: [(0.01, 3), (0.07485, 3), (1.0, 2)], ...}
    source_cluster: the initial cluster that nodes must have at τ = tau_i.
    tau_i: initial τ value; if not specified, the first τ value from an arbitrary node is used.
    tau_f: final τ value; if not specified, the second τ value from an arbitrary node is used.
    tol: tolerance for matching τ values.
    
    Returns a dictionary mapping node IDs that started in source_cluster at tau_i 
    and then switched to a different cluster by tau_f. For each such node, the returned
    information includes the cluster at tau_i, the cluster at tau_f, and the full assignment history.
    """
    # If tau_i or tau_f is not provided, get them from the first node's history.
    if tau_i is None or tau_f is None:
        sample_node = next(iter(partdict_tau))
        sample_history = partdict_tau[sample_node]
        if len(sample_history) >= 2:
            if tau_i is None:
                tau_i = sample_history[0][0]
            if tau_f is None:
                tau_f = sample_history[1][0]
        else:
            raise ValueError("Not enough τ values in node history to set defaults.")

    moved = {}
    for node, history in partdict_tau.items():
        cluster_i = None
        cluster_f = None
        for t, cluster in history:
            if abs(t - tau_i) < tol:
                cluster_i = cluster
            if abs(t - tau_f) < tol:
                cluster_f = cluster
        # Only consider nodes that have assignments at both tau_i and tau_f
        if cluster_i is None or cluster_f is None:
            continue
        # Record node if it started in source_cluster at tau_i and then switched by tau_f
        if cluster_i == source_cluster and cluster_f != source_cluster:
            moved[node] = {"tau_i_cluster": cluster_i,
                           "tau_f_cluster": cluster_f,
                           "history": history}
    return moved
