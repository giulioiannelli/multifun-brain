from .core import *

def generate_hmn(levels=3, base_module_size=4, p_in=1.0, p_out=0.05, seed=None):
    """
    Generate a Hierarchical Modular Network (HMN) using recursive grouping.

    Parameters:
        levels (int): Number of hierarchical levels.
        base_module_size (int): Number of nodes in the lowest-level modules.
        p_in (float): Intra-module connection probability.
        p_out (float): Inter-module connection probability (per level).
        seed (int or None): Random seed.

    Returns:
        G (networkx.Graph): Hierarchical Modular Network.
    """
    rng = np.random.default_rng(seed)

    # Initial modules (level 0)
    modules = []
    node_id = 0
    G = nx.Graph()

    for _ in range(2 ** levels):  # 2^levels modules at bottom
        nodes = list(range(node_id, node_id + base_module_size))
        G.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if i < j and rng.random() < p_in:
                    G.add_edge(i, j)
        modules.append(nodes)
        node_id += base_module_size

    # Recursively build higher-level modules
    current_modules = modules
    for level in range(1, levels + 1):
        next_modules = []
        for i in range(0, len(current_modules), 2):
            if i + 1 >= len(current_modules):
                next_modules.append(current_modules[i])
                continue
            group1 = current_modules[i]
            group2 = current_modules[i + 1]
            next_modules.append(group1 + group2)
            # Add sparse connections between modules
            for u in group1:
                for v in group2:
                    if rng.random() < p_out:
                        G.add_edge(u, v)
        current_modules = next_modules

    return G


def generate_flower_graph(u=2, v=2, iterations=3):
    """
    Generate a (u, v)-flower graph using recursive replacement of edges.

    Parameters:
        u (int): Number of parallel paths of the first type per edge.
        v (int): Number of parallel paths of the second type per edge.
        iterations (int): Number of recursive steps.

    Returns:
        G (networkx.Graph): The resulting flower graph.
    """
    G = nx.Graph()
    G.add_edge(0, 1)
    node_counter = 2

    for _ in range(iterations):
        edges = list(G.edges())
        G.remove_edges_from(edges)

        for u_edge, v_edge in edges:
            # Create u parallel paths
            last = u_edge
            for _ in range(u - 1):
                G.add_node(node_counter)
                G.add_edge(last, node_counter)
                last = node_counter
                node_counter += 1
            G.add_edge(last, v_edge)

            # Create v parallel paths
            last = u_edge
            for _ in range(v - 1):
                G.add_node(node_counter)
                G.add_edge(last, node_counter)
                last = node_counter
                node_counter += 1
            G.add_edge(last, v_edge)

    return G


def generate_brain_timeseries(
    n_regions=5,
    n_timepoints=500,
    sampling_rate=100,  # Hz
    noise_level=0.2,
    seed=None,
    return_time=False
):
    """
    Generate synthetic brain-like time series (e.g., neural regions or sensors).

    Parameters:
        n_regions (int): Number of regions/nodes/sources.
        n_timepoints (int): Number of time points.
        sampling_rate (int): Sampling frequency in Hz.
        noise_level (float): Amount of random noise.
        seed (int or None): Random seed for reproducibility.
        return_time (bool): If True, also return the time vector.

    Returns:
        ts (ndarray): Array of shape (n_regions, n_timepoints).
        time (ndarray): (optional) Time vector in seconds.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
    ts = []

    base_freqs = [5, 10, 20]  # theta, alpha, beta
    for i in range(n_regions):
        signal = np.zeros_like(t)

        # Combine sinusoids with random phase and amplitude
        for f in base_freqs:
            amp = rng.uniform(0.5, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * f * t + phase)

        # Add slow drift
        drift = rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2*np.pi))
        signal += drift

        # Add noise
        signal += rng.normal(0, noise_level, size=t.shape)

        ts.append(signal)

    ts = np.array(ts)

    # Optionally induce correlation (simulate functional connectivity)
    mixing = rng.normal(size=(n_regions, n_regions))
    ts = mixing @ ts

    if return_time:
        return ts, t
    return ts


def generate_brain_timeseries_2(
    n_regions=5,
    region_sizes=4,  # can be int or list
    n_timepoints=500,
    sampling_rate=100,
    noise_level=0.2,
    seed=None,
    return_time=False
):
    """
    Generate synthetic modular brain-like time series with intra-region correlation.

    Parameters:
        n_regions (int): Number of distinct regions/modules.
        region_sizes (int or list): Number of nodes per region. Can be a list of ints or a single int.
        n_timepoints (int): Number of time points.
        sampling_rate (int): Sampling frequency in Hz.
        noise_level (float): Amount of random noise.
        seed (int or None): Random seed for reproducibility.
        return_time (bool): If True, also return the time vector.

    Returns:
        ts (ndarray): Array of shape (total_nodes, n_timepoints).
        time (ndarray): (optional) Time vector in seconds.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
    ts = []

    if isinstance(region_sizes, int):
        region_sizes = [region_sizes] * n_regions
    elif len(region_sizes) != n_regions:
        raise ValueError("Length of region_sizes must equal n_regions if a list is provided.")

    base_freqs = [5, 10, 20]  # theta, alpha, beta

    for size in region_sizes:
        # Shared base signal for the region
        base_signal = np.zeros_like(t)
        for f in base_freqs:
            amp = rng.uniform(0.5, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            base_signal += amp * np.sin(2 * np.pi * f * t + phase)

        # Add slow drift
        drift = rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2*np.pi))
        base_signal += drift

        # Create variations for each node in the region
        for _ in range(size):
            noise = rng.normal(0, noise_level, size=t.shape)
            node_signal = base_signal + noise
            ts.append(node_signal)

    ts = np.array(ts)

    if return_time:
        return ts, t
    return ts


def generate_brain_timeseries_3(
    n_regions=5,
    region_sizes=4,
    n_timepoints=500,
    sampling_rate=100,
    noise_level=0.2,
    seed=None,
    return_time=False
):
    """
    Generate synthetic brain-like time series with modular structure and intra-region correlation.

    Parameters:
        n_regions (int): Number of regions.
        region_sizes (int or list): Nodes per region.
        n_timepoints (int): Time points per signal.
        sampling_rate (int): Hz.
        noise_level (float): Std dev of node-level noise.
        seed (int or None): Random seed.
        return_time (bool): If True, return the time vector.

    Returns:
        ts (ndarray): shape (total_nodes, n_timepoints)
        time (optional): time vector
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    if isinstance(region_sizes, int):
        region_sizes = [region_sizes] * n_regions
    elif len(region_sizes) != n_regions:
        raise ValueError("Length of region_sizes must equal n_regions if a list is provided.")

    all_signals = []

    for size in region_sizes:
        # Shared base: band-limited noise (e.g., 1-30 Hz like EEG)
        raw_noise = rng.normal(0, 1, size=n_timepoints)
        base_signal = bandpass_filter(raw_noise, 1, 30, fs=sampling_rate)

        # Normalize base
        base_signal = (base_signal - np.mean(base_signal)) / np.std(base_signal)

        # Add node-specific variations (low noise to preserve correlation)
        for _ in range(size):
            node_noise = rng.normal(0, noise_level, size=n_timepoints)
            signal = base_signal + node_noise
            all_signals.append(signal)

    ts = np.array(all_signals)

    if return_time:
        return ts, t
    return ts

def generate_brain_timeseries_4(
    n_regions=5,
    region_sizes=4,
    n_timepoints=500,
    sampling_rate=100,
    noise_level=0.2,
    seed=None,
    return_time=False,
    return_labels=False
):
    """
    Generate synthetic brain-like time series with modular structure and intra-region correlation.

    Returns:
        ts (ndarray): shape (total_nodes, n_timepoints)
        time (optional): time vector
        region_labels (optional): array mapping each node to its region
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    if isinstance(region_sizes, int):
        region_sizes = [region_sizes] * n_regions
    elif len(region_sizes) != n_regions:
        raise ValueError("Length of region_sizes must equal n_regions if a list is provided.")

    all_signals = []
    region_labels = []

    for region_idx, size in enumerate(region_sizes):
        # Shared base signal for the region
        raw_noise = rng.normal(0, 1, size=n_timepoints)
        base_signal = bandpass_filter(raw_noise, 1, 30, fs=sampling_rate)
        base_signal = (base_signal - np.mean(base_signal)) / np.std(base_signal)

        for _ in range(size):
            node_noise = rng.normal(0, noise_level, size=n_timepoints)
            signal = base_signal + node_noise
            all_signals.append(signal)
            region_labels.append(region_idx)

    ts = np.array(all_signals)

    outputs = (ts,)
    if return_time:
        outputs += (t,)
    if return_labels:
        outputs += (region_labels,)

    return outputs if len(outputs) > 1 else outputs[0]

def generate_brain_timeseries_5(
    n_regions=5,
    region_sizes=4,
    n_timepoints=500,
    sampling_rate=100,
    noise_level=0.2,
    seed=None,
    return_time=False
):
    """
    Generate synthetic brain-like time series with modular structure and intra-region correlation.

    Parameters:
        n_regions (int): Number of regions.
        region_sizes (int or list): Nodes per region.
        n_timepoints (int): Time points per signal.
        sampling_rate (int): Hz.
        noise_level (float): Std dev of node-level noise.
        seed (int or None): Random seed.
        return_time (bool): If True, return the time vector.

    Returns:
        ts (ndarray): shape (total_nodes, n_timepoints)
        time (optional): time vector
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    if isinstance(region_sizes, int):
        region_sizes = [region_sizes] * n_regions
    elif len(region_sizes) != n_regions:
        raise ValueError("Length of region_sizes must equal n_regions if a list is provided.")

    all_signals = []

    for size in region_sizes:
        # Shared base: band-limited noise (e.g., 1-30 Hz like EEG)
        raw_noise = rng.normal(0, 1, size=n_timepoints)
        base_signal = bandpass_filter(raw_noise, 1, 30, fs=sampling_rate)

        # Normalize base
        base_signal = (base_signal - np.mean(base_signal)) / np.std(base_signal)

        # Add node-specific variations (low noise to preserve correlation)
        for _ in range(size):
            node_noise = rng.normal(0, noise_level, size=n_timepoints)
            signal = base_signal + node_noise
            all_signals.append(signal)

    ts = np.array(all_signals)

    if return_time:
        return ts, t
    return ts

def generate_multiscale_graph(n_clusters=4, cluster_size=10, p_in=0.8, p_out=0.05, seed=None):
    """
    Create a modular graph with multiple structural scales:
    - Dense intra-cluster connectivity
    - Sparse inter-cluster connectivity

    Parameters:
        n_clusters (int): Number of clusters (modules).
        cluster_size (int): Number of nodes per cluster.
        p_in (float): Probability of intra-cluster edges.
        p_out (float): Probability of inter-cluster edges.
        seed (int or None): Random seed.

    Returns:
        G (networkx.Graph): The resulting multiscale graph.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    total_nodes = n_clusters * cluster_size

    # Track node offsets to assign unique IDs
    node_offset = 0
    clusters = []

    for c in range(n_clusters):
        # Create an intra-cluster Erdős–Rényi graph
        cluster = nx.erdos_renyi_graph(cluster_size, p_in, seed=int(rng.integers(1e6)))
        mapping = {node: node + node_offset for node in cluster.nodes()}
        cluster = nx.relabel_nodes(cluster, mapping)
        G = nx.compose(G, cluster)
        clusters.append(list(mapping.values()))
        node_offset += cluster_size

    # Add sparse inter-cluster connections
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            for node_i in clusters[i]:
                for node_j in clusters[j]:
                    if rng.random() < p_out:
                        G.add_edge(node_i, node_j)

    return G


def generate_hierarchical_multiscale_graph(levels=[4, 3, 2], p_in=0.9, p_decay=0.4, seed=None):
    """
    Generate a hierarchical modular graph with multiple structural scales.

    Parameters:
        levels (list of int): Number of groups per level, from lowest to highest scale.
                              E.g., [4, 3, 2] = 4 submodules per module, 3 modules per macro.
        p_in (float): Base probability for intra-group edges at the lowest level.
        p_decay (float): Factor by which edge probability decays at each higher level.
        seed (int or None): Random seed.

    Returns:
        G (networkx.Graph): Graph with hierarchical modular structure.
    """
    rng = np.random.default_rng(seed)
    node_counter = 0
    current_nodes = []
    G = nx.Graph()

    def add_edges_between(group, p):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if rng.random() < p:
                    G.add_edge(group[i], group[j])

    # Bottom-up hierarchical build
    groups = []
    total_nodes = 1
    for l in levels:
        total_nodes *= l

    # Create lowest-level nodes
    current_nodes = list(range(total_nodes))
    G.add_nodes_from(current_nodes)
    groups.append(current_nodes)

    groupings = [current_nodes]
    for level, n_groups in enumerate(levels):
        next_grouping = []
        group_size = len(groupings[-1]) // n_groups

        for i in range(n_groups):
            group = groupings[-1][i * group_size:(i + 1) * group_size]
            add_edges_between(group, p_in * (p_decay ** level))  # decay prob with level
            next_grouping.append(group)

        groupings.append([node for group in next_grouping for node in group])

    return G


def generate_hierarchical_modular_pa(
    base_module_size=10,
    levels=3,
    intra_p=0.8,
    inter_edges=2,
    seed=None
):
    """
    Create a hierarchical modular network with preferential attachment between modules.

    Parameters:
        base_module_size (int): Nodes per base module.
        levels (int): Number of hierarchy levels.
        intra_p (float): Intra-module connection probability.
        inter_edges (int): PA-based inter-module edges per module pair at each level.
        seed (int): Random seed.

    Returns:
        G (networkx.Graph): Final graph with hierarchical modular structure and PA links.
    """
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph()
    node_counter = 0
    modules = []

    # Step 1: Build base modules
    num_modules = 2 ** levels
    for _ in range(num_modules):
        nodes = list(range(node_counter, node_counter + base_module_size))
        G.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if i < j and random.random() < intra_p:
                    G.add_edge(i, j)
        modules.append(nodes)
        node_counter += base_module_size

    # Step 2: Build higher-level modules and connect with PA
    for level in range(levels):
        new_modules = []
        for i in range(0, len(modules), 2):
            if i + 1 >= len(modules):
                new_modules.append(modules[i])
                continue

            group1 = modules[i]
            group2 = modules[i + 1]
            new_group = group1 + group2
            new_modules.append(new_group)

            # Ensure at least one connection (to guarantee connectivity)
            u = random.choice(group1)
            v = random.choice(group2)
            G.add_edge(u, v)

            # Additional preferential attachment connections
            degrees = dict(G.degree(new_group))
            targets = list(degrees.keys())
            probs = np.array([degrees[n] for n in targets], dtype=float)
            probs /= probs.sum()

            attempts = 0
            edges_added = 0
            while edges_added < inter_edges and attempts < 10 * inter_edges:
                u = random.choice(group1)
                v = np.random.choice(targets, p=probs)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    edges_added += 1
                attempts += 1

        modules = new_modules

    return G

def generate_hierarchical_modular_network(num_hierarchies, modules_per_hierarchy, nodes_per_module):
    """
    num_hierarchies: integer, number of hierarchical layers (levels)
    modules_per_hierarchy: list or tuple of length num_hierarchies 
        indicating how many modules each hierarchy should have
    nodes_per_module: list or tuple of length num_hierarchies 
        indicating how many nodes each module should have in that hierarchy

    Returns a NetworkX Graph with a simple hierarchical modular structure:
      1) Each hierarchy h has (modules_per_hierarchy[h]) modules.
      2) Each module is a complete subgraph of (nodes_per_module[h]) new nodes.
      3) Modules in hierarchy h connect to the corresponding module in hierarchy (h+1) by a single edge (example scheme).
    """
    G = nx.Graph()
    current_node_id = 0

    # Keep track of the "module representative" to connect across hierarchies
    # For each hierarchy, store a list of one representative node per module
    representatives = []

    for h in range(num_hierarchies):
        reps_in_this_hierarchy = []
        for mod in range(modules_per_hierarchy[h]):
            module_nodes = range(current_node_id, current_node_id + nodes_per_module[h])
            # Add nodes
            for n in module_nodes:
                G.add_node(n, hierarchy=h, module=mod)
            # Fully connect nodes in the module
            for i in module_nodes:
                for j in module_nodes:
                    if i < j:
                        G.add_edge(i, j)
            # Pick the first node as a "representative" for cross-hierarchy connections
            reps_in_this_hierarchy.append(module_nodes[0])
            current_node_id += nodes_per_module[h]
        representatives.append(reps_in_this_hierarchy)

        # Connect modules of this hierarchy to modules of the previous one (if h>0)
        if h > 0:
            # Example scheme: connect each module to the "same index" module in the previous hierarchy
            for mod_idx, rep in enumerate(reps_in_this_hierarchy):
                prev_rep = representatives[h-1][mod_idx % len(representatives[h-1])]
                G.add_edge(rep, prev_rep)

    return G