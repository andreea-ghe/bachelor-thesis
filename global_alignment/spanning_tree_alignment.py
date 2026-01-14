import numpy as np
from .utils_pose_graph import minimum_spanning_tree


def spanning_tree_alignment(n_nodes, edges, transformations, uncertainties):
    """
    This is a fallback method in case shonan averaging fails to converge. It's simpler
    than Shonan averaging and always succeeds. However, it doesn't optimize globally - 
    errors can accumulate along long paths in the tree.
    Compute global poses by propagating transformations along a minimum spanning tree.
    
    Steps:
    - Build MST from the pose graph using uncertainties as weights
      (lower uncertainty = more reliable edge)
    - Traverse the MST from the anchor node (node 0)
    - For each node, compute its global pose by chaining transformations from the root
    
    Input:
        n_nodes: number of nodes (pieces + 1 virtual node)
        edges: edge list (i, j) pairs: [E, 2]
        transformations: 
            - relative transformation T_ij for each edge: [E, 4, 4]
            - T_ij transforms points from piece j's frame to piece i's frame
        uncertainties: uncertainty weights for each edge (lower = more reliable): [E]
    
    Output:
        global_poses: global pose for each node: [n_nodes, 4, 4]
        success: int - always 1 (this method always succeeds)
    """
    # build MST and get traversal order
    # mst_order: nodes in DFS order starting from node 0
    # mst_predecessors: dict mapping each node to its parent in the tree
    mst_order, mst_predecessors = minimum_spanning_tree(n_nodes, edges, uncertainties)
    
    # create transformation lookup table
    hash_map = np.zeros((n_nodes, n_nodes, 4, 4)) # stores T_ij (transformation from j to i)
    for i in range(edges.shape[0]):
        src, dst = edges[i, 0], edges[i, 1]
        hash_map[src, dst, :, :] = transformations[i, :, :]
        # store both directions for easy lookup during traversal
        hash_map[dst, src, :, :] = np.linalg.inv(transformations[i, :, :])
    
    # initialize global poses
    global_poses = np.zeros((n_nodes, 4, 4))
    global_poses[0, :, :] = np.eye(4) # piece 0 is the anchor with identity pose
    
    # propagate poses along the tree
    for i in range(1, n_nodes): # for each node (except root), compute its global pose from its parent
        child = mst_order[i]
        parent = mst_predecessors[child]
        # global_pose[child] = global_pose[parent] @ T_parent-child
        global_poses[child, :, :] = global_poses[parent, :, :] @ hash_map[parent, child, :, :]
    
    return global_poses, 1  # Always succeeds
