import numpy as np
import itertools
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from .utils_alignment import global_alignment


def global_transform(pred_match_matrix, part_pcs, n_valid, n_pcs, critical_points_idx, n_critical_points, gt_part_rot, gt_part_trans):
    """
    Estimate global transformation using the matching matrix.
    Align to the largest piece as the reference piece.

    Input:
        pred_match_matrix: [B, N_crit_sum, N_crit_sum] - predicted matching matrix between critical points of pieces
        part_pcs: [B, N_pcs_sum, 3] - point clouds of each piece
        n_valid: [B] - number of valid pieces per object
        n_pcs: [B, P] - number of points per piece
        critical_points_idx: [B, N_pcs_sum] - indices of critical points for each piece
        n_critical_points: [B, P] - number of critical points per piece
        gt_part_rot: [B, P, 4] - ground truth rotations (quaternions) of each piece
        gt_part_trans: [B, P, 3] - ground truth translations of each piece
    
    Output:
        pred_dict: dict with keys:
            'rot': [B, P, 3, 3] - estimated rotations (rotation matrices) of each piece
            'trans': [B, P, 3] - estimated translations of each piece
    """
    B, P = n_critical_points.shape

    # compute cumulative sums for indexing
    n_critical_points_cumsum = np.cumsum(n_critical_points, axis=-1)  # [B, P]
    n_pcs_cumsum = np.cumsum(n_pcs, axis=-1)  # [B, P]

    pred_dict = {}
    pred_dict['rot'] = np.zeros((B, P, 3, 3))
    pred_dict['trans'] = np.zeros((B, P, 3))

    for b in range(B):
        piece_connections = np.zeros(n_valid[b]) # number of connections for each piece
        sum_piece_match = np.sum(pred_match_matrix[b]) # total number of matches for all pieces

        edges = []
        transformations = []
        uncertainties = []

        # compute relative poses between all pairs
        for idx1, idx2 in itertools.combinations(np.arange(n_valid[b]), 2):
            # because critical points are concatenated piece by piece, we can use cumulative sums to get the start and end indices
            critical_start1 = 0 if idx1 == 0 else n_critical_points_cumsum[b, idx1 - 1]
            critical_end1 = n_critical_points_cumsum[b, idx1]
            critical_start2 = 0 if idx2 == 0 else n_critical_points_cumsum[b, idx2 - 1]
            critical_end2 = n_critical_points_cumsum[b, idx2]

            # we determine start and end indices for point clouds of the pieces
            points_start1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            points_end1 = n_pcs_cumsum[b, idx1]
            points_start2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            points_end2 = n_pcs_cumsum[b, idx2]

            # determine the nr of critical points for the two pieces
            n1 = n_critical_points[b, idx1]
            n2 = n_critical_points[b, idx2]
            if n1 == 0 or n2 == 0:
                continue

            # extract the sub-matrix for the two pieces
            match_submatrix = pred_match_matrix[b, critical_start1:critical_end1, critical_start2:critical_end2]  # [n1, n2]
            n_matches = np.sum(match_submatrix).astype(np.int32)

            match_sumbatrix2 = pred_match_matrix[b, critical_start2:critical_end2, critical_start1:critical_end1]  # [n2, n1]
            n_matches2 = np.sum(match_sumbatrix2).astype(np.int32)

            if n_matches < n_matches2:
                n_matches = n_matches2
                match_submatrix = match_sumbatrix2.transpose(1, 0)

            if n_valid[b] > 2 and n_matches == 0 and sum_piece_match > 0:
                continue  # skip if no matches and more than 2 pieces

            if n_matches < 3:
                continue  # need at least 3 matches to estimate a 3D transform

            pc_piece1 = part_pcs[b, points_start1:points_end1]  # [N1, 3]
            pc_piece2 = part_pcs[b, points_start2:points_end2]  # [N2, 3]

            if critical_points_idx is not None:
                # use critical points only
                critical_source_points = pc_piece1[critical_points_idx[b, points_start1:points_start1 + n1]] # [n1, 3]
                critical_target_points = pc_piece2[critical_points_idx[b, points_start2:points_start2 + n2]] # [n2, 3]

                edges.append(np.array([idx1, idx2]))
                rigid_transform = estimate_rigid_transform_from_matching(critical_source_points, critical_target_points, match_submatrix)
                transformations.append(rigid_transform)
                uncertainties.append(1 / n_matches)  # uncertainty inversely proportional to number

                # update piece connection counts
                piece_connections[idx1] += 1
                piece_connections[idx2] += 1

        # connect small pieces with less than 3 correspondences
        for idx1, idx2 in itertools.combinations(np.arange(n_valid[b]), 2):
            if piece_connections[idx1] > 0 and piece_connections[idx2] > 0:
                continue # already connected

            if piece_connections[idx1] == 0 and piece_connections[idx2] == 0:
                continue # both unconnected

            # we need to connect pieces to something that is already connected
            critical_start1 = 0 if idx1 == 0 else n_critical_points_cumsum[b, idx1 - 1]
            critical_end1 = n_critical_points_cumsum[b, idx1]
            critical_start2 = 0 if idx2 == 0 else n_critical_points_cumsum[b, idx2 - 1]
            critical_end2 = n_critical_points_cumsum[b, idx2]

            points_start1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            points_end1 = n_pcs_cumsum[b, idx1]
            points_start2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            points_end2 = n_pcs_cumsum[b, idx2]

            n1 = n_critical_points[b, idx1]
            n2 = n_critical_points[b, idx2]

            if n1 == 0 or n2 == 0: # one of the pieces has no critical points
                pc_piece1 = part_pcs[b, points_start1:points_end1]  # [N1, 3]
                pc_piece2 = part_pcs[b, points_start2:points_end2]  # [N2, 3]

                edges.append(np.array([idx1, idx2]))
                rigid_transform = np.eye(4)  # identity transform
                if n2 > 0: # only piece 2 has critical points
                    rigid_transform[:3, 3] = pc_piece2[critical_points_idx[b, points_start2]] - np.sum(pc_piece1, axis=0)
                elif n1 > 0: # only piece 1 has critical points
                    rigid_transform[:3, 3] = np.sum(pc_piece2, axis=0) - pc_piece1[critical_points_idx[b, points_start1]]
                else:
                    rigid_transform[:3, 3] = np.sum(pc_piece2, axis=0) - np.sum(pc_piece1, axis=0)
                transformations.append(rigid_transform)
                uncertainties.append(1)  # high uncertainty

                piece_connections[idx1] += 1
                piece_connections[idx2] += 1
                continue

            match_submatrix = pred_match_matrix[b, critical_start1:critical_end1, critical_start2:critical_end2]  # [n1, n2]
            n_matches = np.sum(match_submatrix).astype(np.int32)

            match_sumbatrix2 = pred_match_matrix[b, critical_start2:critical_end2, critical_start1:critical_end1]  # [n2, n1]
            n_matches2 = np.sum(match_sumbatrix2).astype(np.int32)

            if n_matches < n_matches2:
                n_matches = n_matches2
                match_submatrix = match_sumbatrix2.transpose(1, 0)

            pc_piece1 = part_pcs[b, points_start1:points_end1]  # [N1, 3]
            pc_piece2 = part_pcs[b, points_start2:points_end2]  # [N2, 3]

            if critical_points_idx is not None:
                critical_source_points = pc_piece1[critical_points_idx[b, points_start1:points_start1 + n1]] # [n1, 3]
                critical_target_points = pc_piece2[critical_points_idx[b, points_start2:points_start2 + n2]] # [n2, 3]

                edges.append(np.array([idx1, idx2]))
                rigid_transform = np.eye(4)
                matchin1, matching2 = np.nonzero(match_submatrix)
                rigid_transform[:3, 3] = np.sum(critical_target_points[matching2], axis=0) - np.sum(critical_source_points[matchin1], axis=0)
                transformations.append(rigid_transform)
                uncertainties.append(1)  # high uncertainty

                piece_connections[idx1] += 1
                piece_connections[idx2] += 1

        if len(edges) > 0:
            edges = np.stack(edges)
            transformations = np.stack(transformations)
            uncertainties = np.stack(uncertainties)

            global_transformations = global_alignment(edges, transformations, uncertainties, n_valid[b])
            biggest_piece_idx = 1
            for i in range(n_valid[b]):
                nr_points = n_pcs[b, i]
                if nr_points > n_pcs[b, biggest_piece_idx]: # find the largest piece
                    biggest_piece_idx = i
        else:
            global_transformations = np.repeat(np.eye(4).reshape(1, 4, 4), n_valid[b], axis=0)
            biggest_piece_idx = 0

        global_transformations = align_to_reference_piece(biggest_piece_idx, global_transformations, gt_part_rot[b], gt_part_trans[b], n_valid[b])

        pred_dict['rot'][b, :n_valid[b]] = global_transformations[:, :3, :3]
        pred_dict['trans'][b, :n_valid[b]] = global_transformations[:, :3, 3]

    return pred_dict


def extract_correspondences_from_matching(matching_submatrix):
    """
    Extract correspondences from the matching sub-matrix between two pieces.

    Input:
        matching_submatrix: [n1, n2] - matching scores between critical points of piece 1 and piece 2

    Output:
        correspondences: [n_matches, 2] - indices of matching critical points
    """
    assert isinstance(matching_submatrix, np.ndarray)
    return np.vstack(matching_submatrix.nonzero()).T  # [n_matches, 2]


def estimate_rigid_transform_from_matching(source_points, target_points, matching_submatrix):
    """
    Estimate rigid transformation from source to target using correspondences.

    Input:
        source_points: [N1, 3] - critical points of source piece
        target_points: [N2, 3] - critical points of target piece
        matching_submatrix: [n1, n2] - matching scores between critical points

    Output:
        rigid_transform: dict with 'rot' (3x3 rotation matrix) and 'trans' (3D translation vector)
    """
    correspondences = extract_correspondences_from_matching(matching_submatrix)  # [n_matches, 2]
    rigid_transform = ransac(source_points, target_points, correspondences)
    return rigid_transform


def align_to_reference_piece(ref_idx, global_transformations, gt_part_rot, gt_part_trans, n_valid):
    """
    Align all pieces to the reference piece's ground truth pose.

    Input:
        ref_idx: int - index of the reference piece
        global_transformations: [N, 4, 4] - estimated global transformations for each piece
        gt_part_rot: [P, 4] - ground truth rotations (quaternions) for each piece
        gt_part_trans: [P, 3] - ground truth translations for each piece
        n_valid: int - number of valid pieces

    """
    # construct the ground truth transformation matrix for the reference piece
    align_to_reference = np.eye(4)
    align_to_reference[:3, :3] = R.from_quat(gt_part_rot[ref_idx][[1, 2, 3, 0]]).as_matrix()
    align_to_reference[:3, 3] = gt_part_trans[ref_idx]

    # compute the offset necessary to align the estimated pose of the reference piece to its ground truth pose
    offset = align_to_reference @ np.linalg.inv(global_transformations[ref_idx, :, :])

    # apply offset to all pieces
    for idx in range(n_valid):
        global_transformations[idx, :, :] = offset @ global_transformations[idx, :, :]

    return global_transformations

def ransac(source_points, target_points, correspondences):
    """
    Given noisy point-to-point matches between two pieces, estimate the rigid transform (rotation + translation) 
    that best aligns them, while ignoring wrong matches.
    Random Sample Consensus works like this:
    1. Randomly sample the minimum number of correspondences needed to estimate the transform (3 for 3D).
    2. Estimate the transform from these correspondences.
    3. Apply the transform to all source points
    4. Count how many points are within a certain distance threshold of their corresponding target points.
    5. Repeat for a fixed number of iterations and keep the transform with the most inliers (smallest error).

    Input:
        source_points: [N1, 3] - critical points of source piece
        target_points: [N2, 3] - critical points of target piece
        correspondences: [n_matches, 2] - indices of matching critical points

    Output:
        transformation: [4, 4] - estimated rigid transformation matrix
    """
    distance_threshold = 0.05

    # convert to open3d PointCloud object
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    # convert correspondences to open3d format
    corres = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=source,
        target=target,
        corres=corres,
        max_correspondence_distance=distance_threshold, # inlier threshold
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False), # point-to-point estimation without scaling
        ransac_n=3, # 3 randomly chosen correspondences to estimate each hypothesis
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 2500) # max iterations and confidence
    )

    return result.transformation  # [4, 4]


if __name__ == "__main__":
    n = 400
    pc1 = np.random.random((n, 3))
    pc2 = pc1.copy()

    transform_matrix = np.zeros((4, 4))
    transform_matrix[:3, :3] = R.random().as_matrix()
    transform_matrix[:3, 3] = np.random.random(3)
    transform_matrix[3, 3] = 1

    pc1 = np.concatenate([pc1, np.ones((pc1.shape[0], 1))], axis=1)  # [N, 4]
    pc1 = transform_matrix @ pc1.transpose(1, 0)
    pc1 = pc1[:3, :].transpose(1, 0)  # [N, 3]

    print(transform_matrix)
