from utilities.rotation import Rotation3D
from pytorch3d.transforms import quaternion_apply


def transform_point_cloud(point_cloud, translation, rotation):
    """
    Returns the transformed point cloud given translation and rotation.
    """
    assert isinstance(rotation, Rotation3D)
    rot = rotation.rot
    rot_type = rotation.rot_type

    if rot_type == 'quat':
        return quat_transform(point_cloud, translation, rot)
    elif rot_type == 'rmat':
        return rmat_transform(point_cloud, translation, rot)
    else:
        raise NotImplementedError(f"Rotation type {rot_type} not implemented.")

def quat_transform(point_cloud, translation, quaternion):
    """
    Rotate vector(s) point_cloud about the rotation described by quaternion(s) 
    and then translate.

    Input:
        point_cloud: [B, P, N, 3] - original point clouds
        translation: [B, P, 3] - translations
        quaternion: [B, P, 4] - quaternions

    Output:
        transformed_pc: [B, P, N, 3] - transformed point clouds
    """
    assert translation.shape[-1] == 3
    assert quaternion.shape[-1] == 4

    if len(translation.shape) == len(point_cloud.shape) - 1:
        translation = translation.unsqueeze(-2).repeat_interleave(point_cloud.shape[-2], dim=-2)

    assert translation.shape == point_cloud.shape

    quat_pc = quat_rot(point_cloud, quaternion)  # [B, P, N, 3]
    transformed_pc = quat_pc + translation  # [B, P, N, 3]

    return transformed_pc

def quat_rot(point_cloud, quaternion):
    """
    Rotate vector(s) point_cloud about the rotation described by quaternion(s).

    Input:
        point_cloud: [B, P, N, 3] - original point clouds
        quaternion: [B, P, 4] - quaternions

    Output:
        rotated_pc: [B, P, N, 3] - rotated point clouds
    """
    if len(quaternion.shape) == len(point_cloud.shape) - 1:
        quaternion = quaternion.unsqueeze(-2).repeat_interleave(point_cloud.shape[-2], dim=-2)

    assert quaternion.shape == point_cloud.shape[:-1]

    return quaternion_apply(quaternion, point_cloud)  # [B, P, N, 3]

def rmat_transform(point_cloud, translation, rotation_matrix):
    """
    Rotate vector(s) point_cloud about the rotation described by rotation_matrix(s)
    and then translate.

    Input:
        point_cloud: [B, P, N, 3] - original point clouds
        translation: [B, P, 3] - translations
        rotation_matrix: [B, P, 3, 3] - rotation matrices

    Output:
        transformed_pc: [B, P, N, 3] - transformed point clouds
    """
    assert translation.shape[-1] == 3

    if len(translation.shape) == len(point_cloud.shape) - 1:
        translation = translation.unsqueeze(-2).repeat_interleave(point_cloud.shape[-2], dim=-2)

    assert translation.shape == point_cloud.shape

    rmat = rmat_rot(point_cloud, rotation_matrix)  # [B, P, N, 3]
    transformed_pc = rmat + translation

    return transformed_pc

def rmat_rot(point_cloud, rotation_matrix):
    """
    Rotate vector(s) point_cloud about the rotation described by rotation_matrix(s).
    """
    assert point_cloud.shape[-1] == 3
    assert rotation_matrix.shape[-1] == rotation_matrix.shape[-2] == 3
    
    if len(rotation_matrix.shape) == len(point_cloud.shape) - 1:
        rotation_matrix = rotation_matrix.unsqueeze(-3).repeat_interleave(point_cloud.shape[-2], dim=-3)

    assert rotation_matrix.shape[:-2] == point_cloud.shape[:-1]

    rotated_pc = (rotation_matrix @ point_cloud.unsqueeze(-1)).squeeze(-1)  # [B, P, N, 3]
    return rotated_pc