"""
Global Alignment Module - Recovers absolute poses from relative pose estimates.

This module takes pairwise relative transformations between pieces (estimated
via RANSAC on fracture surface correspondences) and computes globally consistent
absolute poses for all pieces.

Pipeline:
1. Connect disconnected components via a virtual hub node
2. Try Shonan Averaging for optimal rotation synchronization
3. If Shonan fails, fall back to Spanning Tree alignment
4. Canonicalize poses relative to piece 0

The virtual hub node (index = n_valid) connects all components with high-uncertainty
edges, ensuring the pose graph is connected while minimizing influence on the solution.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .utils_shonan import shonan_averaging
from .utils_pose_graph import connect_graph
from .spanning_tree_alignment import spanning_tree_alignment


def global_alignment(edges, transformations, uncertainties, n_valid):
    """
    Perform global alignment using Shonan Averaging with spanning tree fallback.

    Input:
        edges: [E, 2] - edges between pieces (from RANSAC matches)
        transformations: [E, 4, 4] - relative transformations for each edge
            T_ij transforms points from piece j's frame to piece i's frame
        uncertainties: [E] - uncertainty for each edge (lower = more reliable)
        n_valid: int - number of valid pieces in this assembly

    Output:
        global_poses: [n_valid, 4, 4] - global pose for each piece,
            canonicalized so piece 0 has identity pose
    """
    # Step 1: Add auxiliary edges to connect disconnected components
    # The virtual hub node (index n_valid) connects to one node in each component
    auxiliary_edges = connect_graph(n_valid, edges)
    all_edges = np.concatenate([edges, auxiliary_edges], axis=0).astype(np.int32)

    # Step 2: Create random transformations for auxiliary edges
    # These have high uncertainty so they don't distort the solution
    auxiliary_transformations = []
    for i in range(auxiliary_edges.shape[0]):
        transformation = np.eye(4)
        transformation[:3, :3] = R.random().as_matrix()
        transformation[:3, 3] = np.random.random(3)
        auxiliary_transformations.append(transformation)
    auxiliary_transformations = np.stack(auxiliary_transformations)
    all_transformations = np.concatenate([transformations, auxiliary_transformations], axis=0)

    # Step 3: Set high uncertainty for auxiliary edges
    auxiliary_uncertainties = np.ones(auxiliary_edges.shape[0])  # High uncertainty
    all_uncertainties = np.concatenate([uncertainties, auxiliary_uncertainties], axis=0)

    # Step 4: Try Shonan Averaging first (optimal when it converges)
    n_nodes = n_valid + 1  # Include virtual hub node
    global_pose_results, success = shonan_averaging(
        all_edges, all_transformations, all_uncertainties, n_nodes
    )

    # Step 5: Fall back to Spanning Tree if Shonan fails
    if success == 0:
        global_pose_results, _ = spanning_tree_alignment(
            n_nodes, all_edges, all_transformations, all_uncertainties
        )

    # Step 6: Canonicalize poses relative to piece 0
    # Transform all poses so piece 0 has identity pose
    # This is done by left-multiplying all poses by inv(pose_0)
    pose_0_inv = np.linalg.inv(global_pose_results[0, :, :])
    for i in range(n_valid):
        # Process in reverse order to avoid overwriting pose_0 before we're done
        idx = n_valid - i - 1
        global_pose_results[idx, :, :] = pose_0_inv @ global_pose_results[idx, :, :]

    # Return only the piece poses (exclude virtual hub node)
    return global_pose_results[:n_valid, :, :]
