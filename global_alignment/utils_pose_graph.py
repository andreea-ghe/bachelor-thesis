"""
Pose Graph Utilities - Graph operations for global alignment.

The pose graph represents pieces as nodes and relative pose estimates as edges.
These utilities help ensure the graph is connected (required for global alignment)
and find minimum spanning trees for the fallback alignment method.
"""
import numpy as np
import networkx as nx


def connect_graph(n_valid, edges):
    """
    Ensure the pose graph is connected by adding a virtual hub node.
    
    Problem: RANSAC may fail to find correspondences between some piece pairs,
    leaving the pose graph disconnected (multiple components). Shonan Averaging
    requires a connected graph.
    
    Solution: Add a virtual "hub" node (index = n_valid) that connects to one
    node in each connected component. The hub acts as a central reference point.
    
    The auxiliary edges have high uncertainty, so they don't distort the solution
    but ensure all pieces can be aligned to a common reference frame.
    
    Input:
        n_valid: int - number of valid pieces
        edges: [E, 2] - existing edges from successful RANSAC matches
    
    Output:
        auxiliary_edges: [C, 2] - new edges connecting hub to each component
            where C is the number of connected components
    """
    # Build graph from existing edges
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_valid))
    G.add_edges_from(edges)

    # Find connected components (sorted by size, largest first)
    components = [
        list(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]

    # Connect hub (node n_valid) to one representative from each component
    auxiliary_edges = []
    for component in components:
        # Connect to first node in component (arbitrary choice)
        auxiliary_edges.append([n_valid, component[0]])

    return np.stack(auxiliary_edges)


def minimum_spanning_tree(n_nodes, edges, weights):
    """
    Compute minimum spanning tree and return DFS traversal order.
    
    The MST uses edge weights (uncertainties) to select the most reliable
    edges for the tree. Lower weight = lower uncertainty = more reliable.
    
    Input:
        n_nodes: int - number of nodes in the graph
        edges: [E, 2] - edge list
        weights: [E] - edge weights (uncertainties)
    
    Output:
        dfs_order: list - nodes in DFS preorder starting from node 0
        predecessors: dict - maps each node to its parent in the DFS tree
    """
    # Build weighted graph
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_nodes))
    for i in range(edges.shape[0]):
        G.add_edge(edges[i, 0], edges[i, 1], weight=weights[i])
    
    # Compute MST
    T = nx.minimum_spanning_tree(G)
    
    # Return DFS traversal from node 0
    dfs_order = list(nx.dfs_preorder_nodes(T, source=0))
    predecessors = nx.dfs_predecessors(T, source=0)
    
    return dfs_order, predecessors
