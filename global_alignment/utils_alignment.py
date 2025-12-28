import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils_shonan import shonan_averaging
from .utils_pose_graph import connect_graph


def global_alignment(edges, transformations, uncertainties, n_valid):
    """
    Perform global alignment using Shonan Averaging with auxiliary edges to connect components.

    Input:
        edges: [E, 2] - edges between pieces
        transformations: [E, 4, 4] - relative transformations for each edge
        uncertainties: [E] - uncertainties for each edge
        n_valid: int - number of valid pieces

    Output:
        global_pose_results: [n_valid, 4, 4] - global poses for each piece
    """
    auxiliary_edges = connect_graph(n_valid, edges)
    all_edges = np.concatenate([edges, auxiliary_edges], axis=0).astype(np.int32)

    auxiliary_transformations = []
    for i in range(auxiliary_edges.shape[0]):
        transformation = np.eye(4)
        transformation[:3, :3] = R.random().as_matrix()
        transformation[:3, 3] = np.random.random(3)
        auxiliary_transformations.append(transformation)
    auxiliary_transformations = np.stack(auxiliary_transformations)
    all_transformations = np.concatenate([transformations, auxiliary_transformations], axis=0)

    auxiliary_uncertainties = 1 * np.ones((auxiliary_edges.shape[0]))
    all_uncertainties = np.concatenate([uncertainties, auxiliary_uncertainties], axis=0)

    global_pose_results = shonan_averaging(all_edges, all_transformations, all_uncertainties, n_valid + 1) # we add the virtual node as well
    # canonicalization: express all poses in the coordinate frame of piece 0
    for idx in range(n_valid):
        global_pose_results[n_valid - i - 1, :, :] = np.linalg.inv(global_pose_results[0, :, :]) @ global_pose_results[n_valid - i - 1, :, :]