from .core import *


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

