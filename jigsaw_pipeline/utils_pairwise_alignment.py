import numpy as np
from scipy.spatial.transform import Rotation as R


def pairwise_alignment(critical_pcs_source, critical_pcs_target, match_submatrix):
    """
    We compute the best rotation R and translation t that aligns a source fracture surface to a target 
    fracture surface, using a soft correspondence matrix as weights.

    Input:
        critical_pcs_source: [n1, 3] - source fracture points
        critical_pcs_target: [n2, 3] - target fracture points
        match_submatrix: [n1, n2] - soft correspondence matrix between source and target points

    Output:
        R: [3, 3] - rotation matrix
        t: [3] - translation vector
    """
    # matrix multiplications later are written in linear algebra form, so we transpose the point clouds here
    critical_pcs_source = critical_pcs_source.transpose(1, 0)
    critical_pcs_target = critical_pcs_target.transpose(1, 0)

    # rigid alignment must remove translation first so we compute centroids
    center_critical_pcs_source = critical_pcs_source.mean(axis=1)
    center_critical_pcs_target = critical_pcs_target.mean(axis=1)

    # both point clouds are centered at the origin, and this ensures rotation is solved independently of translation
    critical_pcs_source -= center_critical_pcs_source.reshape(-1, 1)
    critical_pcs_target -= center_critical_pcs_target.reshape(-1, 1)

    # we build the weighted cross-covariance matrix: encodes how the two point clouds correlate, weighted by matching confidence
    M = critical_pcs_source @ match_submatrix @ critical_pcs_target.T
    # convert covariance into quaternion problem (Horn’s method)
    # the optimal rotation corresponds to the largest eigenvector of N
    N = np.array(
        [
            [
                M[0, 0] + M[1, 1] + M[2, 2],
                M[1, 2] - M[2, 1],
                M[2, 0] - M[0, 2],
                M[0, 1] - M[1, 0],
            ],
            [
                M[1, 2] - M[2, 1],
                M[0, 0] - M[1, 1] - M[2, 2],
                M[0, 1] + M[1, 0],
                M[0, 2] + M[2, 0],
            ],
            [
                M[2, 0] - M[0, 2],
                M[0, 1] + M[1, 0],
                M[1, 1] - M[0, 0] - M[2, 2],
                M[1, 2] + M[2, 1],
            ],
            [
                M[0, 1] - M[1, 0],
                M[2, 0] + M[0, 2],
                M[1, 2] + M[2, 1],
                M[2, 2] - M[0, 0] - M[1, 1],
            ],
        ]
    )
    # eigen decomposition
    v, u = np.linalg.eigh(N)
    id = v.argmax()

    # convert quaternion → rotation matrix
    q = u[:, id]
    R = np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[2] * q[1] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[3] * q[1] - q[0] * q[2]),
                2 * (q[3] * q[2] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )

    # restore original coordinates
    critical_pcs_source += center_critical_pcs_source.reshape(-1, 1)
    critical_pcs_target += center_critical_pcs_target.reshape(-1, 1)

    # compute translation
    t = (match_submatrix @ critical_pcs_target.T).T - (np.sum(match_submatrix, axis=-1).reshape((-1, 1)) * (R @ critical_pcs_source).T).T
    t = np.sum(t, axis=-1) / np.sum(match_submatrix)

    return R.astype(np.float32), t.astype(np.float32)


if __name__ == "__main__":
    a = np.random.random((10, 3))

    rot = R.random().as_matrix()
    t = np.random.random(3)
    b = (rot @ a.T).T + t
    weight = np.eye(10) * 0.1

    print(rot, t)
    print(pairwise_alignment(a, b, weight))