import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as R


def shonan_averaging(edges, transformations, uncertainties, n_valids):
    """
    Recover global poses from noisy relative transformations using Shonan Averaging.
    
    Shonan Averaging is a certifiably optimal algorithm for rotation averaging.
    It lifts rotations to higher-dimensional space and solves a semidefinite
    relaxation, providing globally optimal solutions when the relaxation is tight.
    
    Reference: Dellaert et al., "Shonan Rotation Averaging" (ECCV 2020)
    
    Input:
        edges: [E, 2] - edge list (i, j) pairs
        transformations: [E, 4, 4] - relative transformation T_ij for each edge
        uncertainties: [E] - uncertainty weights for each edge
        n_valids: int - number of nodes (pieces + 1 virtual node)
    
    Output:
        global_poses: [n_valids, 4, 4] - estimated global pose for each node
        success: int - 1 if converged, 0 if failed (use spanning tree fallback)
    """
    n_edges = edges.shape[0]
    
    pose_graph_factors = []
    new_uncertainties = [] # uncertainties for the factors we actually add (skip self-loops)
    
    for i in range(n_edges):
        if edges[i, 1] == edges[i, 0]: # skip self-loop edges
            continue

        # add gaussian noise: small variances for high certainty
        pose_noise = gtsam.noiseModel.Diagonal.Variances(uncertainties[i] * np.array([1e-2]*6))
        # each factor encodes a relative pose measurement between two parts, an error that penalizes 
        # deviation from the measured relative pose
        relative_pose_factor = gtsam.BetweenFactorPose3(
            edges[i, 0],
            edges[i, 1],
            gtsam.Pose3(transformations[i, :, :]),
            pose_noise
        )

        pose_graph_factors.append(relative_pose_factor)
        new_uncertainties.append(uncertainties[i])

    # we start randomly because we don't have any prior on the absolute poses
    shonan = gtsam.ShonanAveraging3(gtsam.BetweenFactorPose3s(pose_graph_factors))
    initial = shonan.initializeRandomly()

    try:
        rotations, _ = shonan.run(initial, 3, 10) # we ignore translations, lift rotations to higher dimensional 
        # space and solve a non-convex rotation synchronization problem
        # once we have rotations, we can estimate translations using least squares
        poses = estimate_poses_from_rotations(pose_graph_factors, rotations, np.array(new_uncertainties), d=3)
    
        global_pose_results = []
        for idx in range(poses.size()):
            global_pose_result = poses.atPose3(idx).matrix()
            global_pose_results.append(global_pose_result)

        return np.stack(global_pose_results), 1  # Success
    except:
        # Shonan failed - return identity poses and signal failure
        # The caller should use spanning tree alignment as fallback
        print("Shonan Averaging didn't converge.")

        global_pose_results = []
        for idx in range(n_valids):
            global_pose_result = np.eye(4)
            global_pose_results.append(global_pose_result)
        
        return np.stack(global_pose_results), 0  # Failure - use fallback


def estimate_poses_from_rotations(factors, rotations, uncertainties, d=3):
    """
    Given rotations estimated from Shonan Averaging, estimate the full poses using least squares.
    
    Input:
        factors: list of gtsam BetweenFactorPose3 - relative pose measurements between parts
        rotations: gtsam.Rot3 or gtsam.Rot2 - estimated rotations for each part
        uncertainties: [E] - uncertainties for each relative pose measurement
        d: int - dimension (2 or 3)
    
    Output:
        result_poses: gtsam.Values - estimated full poses for each part
    """
    def get_rotation_of_node(j):
        return rotations.atRot3(j) if d == 3 else rotations.atRot2(j)

    def build_pose(R, t):
        return gtsam.Pose3(R, t) if d == 3 else gtsam.Pose2(R, t)

    identity = np.eye(d)

    graph = gtsam.GaussianFactorGraph() # linear factor graph
    model = gtsam.noiseModel.Unit.Create(d)

    # canonicalization again: anchor the first pose to identity
    graph.add(0, identity, np.zeros((d,)), model) # anchor the first pose

    for idx in range(len(factors)): # for each relative pose measurement between i and j
        factor = factors[idx]
        i, j = factor.keys() # get the indices of the two parts
        Tij = factor.measured() # relative transformation from i to j

        if i == j: # skip self-loop edges
            continue

        pose_model = gtsam.noiseModel.Diagonal.Variances(uncertainties[idx] * np.array([1e-2]*d))
        measured = get_rotation_of_node(i).rotate(Tij.translation())
        graph.add(j, identity, i, -identity, measured, pose_model) # add a linear constraint, a factor encoding the relative translation measurement

    translations = graph.optimize() # solve for translations using least squares
    
    result_poses = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result_poses.insert(j, build_pose(get_rotation_of_node(j), tj))

    return result_poses


if __name__ == "__main__":
    total_num = 3
    
    global_poses = []
    for i in range(total_num):
        transformation = np.eye(4)
        transformation[:3, :3] = R.random().as_matrix()
        transformation[:3, 3] = np.random.rand(3)
        global_poses.append(transformation)
    global_poses = np.stack(global_poses)
    
    n = global_poses.shape[0]
    uncertainty = []
    for i in range(n):
        global_poses[n - i - 1, :, :] = (np.linalg.inv(global_poses[0, :, :]) @ global_poses[n - i - 1, :, :])
    edges = np.stack(
        [
            np.repeat(np.arange(total_num), total_num),
            np.tile(np.arange(total_num), total_num),
        ]
    ).transpose()

    transformations = []
    for i in range(edges.shape[0]):
        transformations.append(np.linalg.inv(global_poses[edges[i, 0], :, :]) @ global_poses[edges[i, 1], :, :])
        uncertainty.append(1)

    for i in range(total_num):
        edges = np.concatenate([edges, np.array([[total_num, i]])], axis=0)
        transformations.append(np.eye(4))
        uncertainty.append(1e6)

    uncertainty = np.array(uncertainty)
    transformations = np.stack(transformations)
    uncertainty[0] = 0.01
    uncertainty[1] = 0.01
    transformations[0, :, :] = np.array(
        [
            [0.58883767, 0.76988153, -0.24607443, 0.20378149],
            [-0.64633755, 0.26572036, -0.71529048, 0.46243953],
            [-0.48530194, 0.58023712, 0.6540695, 0.18572326],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformations[1, :, :] = np.array(
        [
            [0.65344216, -0.75661764, -0.02330413, 0.80860915],
            [-0.74655571, -0.64923189, 0.14543908, 0.18233219],
            [-0.12517156, -0.0776382, -0.98909271, 0.63350485],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    global_pose_results, success = shonan_averaging(edges, transformations, uncertainty, total_num)
    print(f"Shonan converged: {success == 1}")
    for i in range(n):
        global_pose_results[n - i - 1, :, :] = (np.linalg.inv(global_pose_results[0, :, :]) @ global_pose_results[n - i - 1, :, :])

    for i in range(n):
        np.set_printoptions(precision=3, suppress=True)
        print(global_poses[i, :, :])
        np.set_printoptions(precision=3, suppress=True)
        print(global_pose_results[i, :, :])
