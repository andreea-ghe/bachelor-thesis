import numpy as np
import networkx as nx


def connect_graph(n_valid, edges):
    """
    Connect the graph by adding auxiliary edges between connected components.
    """
    # create a graph with nodes being the valid pieces and edges being the relative pose estimates from RANSAC
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_valid))
    G.add_edges_from(edges)

    # find connected components, each component is a subassembly
    components = [
        list(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]

    # add auxiliary edges between the largest component and other components so we don't have a disconnected graph
    # later, these auxiliary edges will have high uncertainty to minimize their influence
    auxiliary_edges = [
        [n_valid, component[0]] for component in components # n_valid is a virtual node so we don't really connect components
    ]

    return np.stack(auxiliary_edges)