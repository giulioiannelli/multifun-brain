from ..multifunbrain import *

# Example usage:
if __name__ == "__main__":
    # Suppose 3 hierarchical layers,
    #   in the first layer: 2 modules, each with 5 nodes,
    #   second layer: 3 modules, each with 4 nodes,
    #   third layer: 2 modules, each with 6 nodes
    net = generate_hierarchical_modular_network(
        num_hierarchies=5,
        modules_per_hierarchy=[5, 3, 2, 7, 1],
        nodes_per_module=[10, 7, 10, 10, 5]
    )
    print("Number of nodes:", net.number_of_nodes())
    print("Number of edges:", net.number_of_edges())
    nx.draw(net); plt.show()
