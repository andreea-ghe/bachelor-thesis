"""
Spanning Tree Alignment - Fallback method for global pose estimation.

When Shonan Averaging fails to converge (common with noisy measurements or 
complex configurations), this method provides a robust alternative using
Minimum Spanning Tree (MST) traversal.

Algorithm:
1. Build MST from the pose graph using edge uncertainties as weights
   (lower uncertainty = more reliable = preferred edge)
2. Traverse the MST starting from node 0 (anchor piece)
3. Propagate poses along tree edges using relative transformations

This is simpler than Shonan but always succeeds. However, it doesn't 
optimize globally - errors can accumulate along long paths in the tree.
"""
import numpy as np

from .utils_pose_graph import minimum_spanning_tree


def spanning_tree_alignment(n_nodes, edges, transformations, uncertainties):
    """
    Compute global poses by propagating transformations along a minimum spanning tree.
    
    The MST is built using uncertainties as edge weights, so the most reliable
    relative pose estimates form the tree backbone.
    
    Input:
        n_nodes: int - number of nodes (pieces + 1 virtual node)
        edges: [E, 2] - edge list (i, j) pairs
        transformations: [E, 4, 4] - relative transformation T_ij for each edge
            T_ij transforms points from piece j's frame to piece i's frame
        uncertainties: [E] - uncertainty weights for each edge (lower = more reliable)
    
    Output:
        global_poses: [n_nodes, 4, 4] - global pose for each node
        success: int - always 1 (this method always succeeds)
    """
    # Step 1: Build MST and get traversal order
    # mst_order: nodes in DFS order starting from node 0
    # mst_predecessors: dict mapping each node to its parent in the tree
    mst_order, mst_predecessors = minimum_spanning_tree(n_nodes, edges, uncertainties)
    
    # Step 2: Create transformation lookup table
    # hash_map[i, j] stores T_ij (transformation from j to i)
    # We store both directions for easy lookup during traversal
    hash_map = np.zeros((n_nodes, n_nodes, 4, 4))
    for i in range(edges.shape[0]):
        src, dst = edges[i, 0], edges[i, 1]
        hash_map[src, dst, :, :] = transformations[i, :, :]
        hash_map[dst, src, :, :] = np.linalg.inv(transformations[i, :, :])
    
    # Step 3: Initialize global poses
    # Node 0 is the anchor with identity pose
    global_poses = np.zeros((n_nodes, 4, 4))
    global_poses[0, :, :] = np.eye(4)
    
    # Step 4: Propagate poses along the tree
    # For each node (except root), compute its global pose from its parent
    # global_pose[child] = global_pose[parent] @ T_parent_to_child
    for i in range(1, n_nodes):
        child = mst_order[i]
        parent = mst_predecessors[child]
        global_poses[child, :, :] = global_poses[parent, :, :] @ hash_map[parent, child, :, :]
    
    return global_poses, 1  # Always succeeds
