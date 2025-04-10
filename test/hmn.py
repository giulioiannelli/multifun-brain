import networkx as nx
import random

def create_modular_network(num_top_level_modules, nodes_per_module,
                           inter_module_edge_prob, intra_module_edge_prob,
                           hierarchy_levels=2, module_splitting_factor=3,
                           recursive_inter_module_prob_multiplier=0.5,
                           seed=None):
    """
    Generates a hierarchical modular network using NetworkX.

    Args:
        num_top_level_modules (int): The number of top-level modules.
        nodes_per_module (int): The approximate number of nodes in each
                                 base-level module.
        inter_module_edge_prob (float): The probability of an edge between nodes
                                        in different modules at the current level.
        intra_module_edge_prob (float): The probability of an edge between nodes
                                        within the same module at the current level.
        hierarchy_levels (int): The number of levels in the hierarchy (>= 2).
        module_splitting_factor (int): The number of sub-modules each module
                                       is split into at the next level.
        recursive_inter_module_prob_multiplier (float): Multiplier for the
                                                        inter-module edge probability
                                                        at deeper levels.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        networkx.Graph: The generated hierarchical modular network.
        dict: A dictionary where keys are nodes and values are their module
              membership at the base level.
    """

    if seed is not None:
        random.seed(seed)

    graph = nx.Graph()
    base_level_modules = {}
    next_module_id = 0

    def _create_module(module_id, nodes, level):
        nonlocal next_module_id
        module_nodes = list(nodes)
        n_nodes = len(module_nodes)

        # Assign module ID to nodes at the current level
        current_module_str = str(module_id)
        for node in module_nodes:
            graph.add_node(node, module=current_module_str) # Ensure node exists with module attribute

        # Add intra-module edges
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < intra_module_edge_prob:
                    graph.add_edge(module_nodes[i], module_nodes[j])

        if level < hierarchy_levels:
            num_sub_modules = module_splitting_factor
            sub_module_size = n_nodes // num_sub_modules
            remaining_nodes = n_nodes % num_sub_modules
            start_index = 0

            for i in range(num_sub_modules):
                end_index = start_index + sub_module_size + (1 if i < remaining_nodes else 0)
                sub_module_nodes = module_nodes[start_index:end_index]
                sub_module_id = f"{module_id}-{i}" # More explicit sub-module ID
                next_module_id += 1
                _create_module(sub_module_id, sub_module_nodes, level + 1)
                start_index = end_index

            # Add inter-module edges between sub-modules of the current module
            current_sub_modules = [m for n, m in graph.nodes(data='module') if m is not None and m.startswith(f"{module_id}-")]
            for i in range(len(current_sub_modules)):
                for j in range(i + 1, len(current_sub_modules)):
                    nodes_i = [n for n, m in graph.nodes(data='module') if m == current_sub_modules[i]]
                    nodes_j = [n for n, m in graph.nodes(data='module') if m == current_sub_modules[j]]
                    current_inter_module_prob = inter_module_edge_prob * (recursive_inter_module_prob_multiplier ** (level - 1))
                    for node_i in nodes_i:
                        for node_j in nodes_j:
                            if random.random() < current_inter_module_prob:
                                graph.add_edge(node_i, node_j)
        else:
            # Base level: Assign final module ID (already done at the beginning of the function)
            for node in module_nodes:
                base_level_modules[node] = current_module_str

    # Create top-level modules
    all_nodes = list(range(num_top_level_modules * nodes_per_module))
    random.shuffle(all_nodes)
    start_index = 0
    for i in range(num_top_level_modules):
        end_index = start_index + nodes_per_module
        module_nodes = all_nodes[start_index:end_index]
        _create_module(i, module_nodes, 1)
        start_index = end_index

    return graph, base_level_modules

if __name__ == "__main__":
    # Example usage:
    num_top_modules = 4
    nodes_per_mod = 30
    inter_prob = 0.3
    intra_prob = 0.05
    hierarchy = 5
    splitting = 4
    recursive_multiplier = 0.3
    random_seed = 42

    hierarchical_graph, module_assignment = create_modular_network(
        num_top_modules, nodes_per_mod, inter_prob, intra_prob,
        hierarchy, splitting, recursive_multiplier, seed=random_seed
    )

    print(f"Number of nodes: {hierarchical_graph.number_of_nodes()}")
    print(f"Number of edges: {hierarchical_graph.number_of_edges()}")
    print(f"Number of base-level modules: {len(set(module_assignment.values()))}")

    # You can further analyze or visualize the graph using NetworkX functions
    # For example, to get the module of a specific node:
    # print(f"Module of node 10: {module_assignment.get(10)}")

    import matplotlib.pyplot as plt
    nx.draw(hierarchical_graph, with_labels=True, node_size=20)
    plt.show()