import torch
import numpy as np

from pytorch3d.transforms import quaternion_multiply
from pytorch3d.transforms import rotation_6d_to_matrix as rot6d_to_matrix

epsilon = 1e-6

class Rotation3D:
    """
    Class for different 3D rotation representations.
    """
    ROT_TYPE = ["quat", "rmat", "axis"]
    ROT_NAME = {
        "quat": "quaternion",
        "rmat": "matrix",
        "axis": "axis_angle"
    }

    def __init__(self, rot, rot_type="quat"):
        self._rot = rot
        self._rot_type = rot_type

        self._check_valid()

    def _process_zero_quat(self):
        """
        Convert zero-norm quaternions to (1, 0, 0, 0) to avoid conversion errors.
        """
        with torch.no_grad():
            norms = torch.norm(self._rot, p=2, dim=-1, keepdim=True)  # [..., 1]
            new_rot = torch.zeros_like(self._rot)
            new_rot[..., 0] = 1.0 # zero quat
            valid_mask = (norms.abs() > 0.5).repeat_interleave(4, dim=-1)
        self._rot = torch.where(valid_mask, self._rot, new_rot)

    def _check_valid(self):
        """
        Check the shape of rotation.
        """
        assert self._rot_type in self.ROT_TYPE, f"Rotation {self._rot_type} not supported."
        assert isinstance(self._rot, torch.Tensor), "Rotation must be a torch Tensor."

        # We always make rotation in float32, otherwise quat won't be unit,
        # and rmat won't be orthogonal.
        self._rot = self._rot.float()

        if self._rot_type == "quat":
            assert self._rot.shape[-1] == 4, "Quaternion must have shape [..., 4]"

            # quat with norm == 0 are padded, make them (1, 0, 0, 0)
            # because (0, 0, 0, 0) convert to rmat will cause PyTorch error.
            self._process_zero_quat()
        
        elif self._rot_type == "rmat":
            if self._rot.shape[-1] == 3:
                if self._rot.shape[-2] == 3: # 3x3 matrix
                    pass
                elif self._rot.shape[-2] == 2: # 6D representation
                    self._rot = rot6d_to_matrix(self._rot.flatten(-2, -1))
                else:
                    raise ValueError("Rotation matrix must have shape [..., 3, 3] or [..., 2, 3]")
            elif self._rot.shape[-1] == 6: # 6D representation
                self._rot = rot6d_to_matrix(self._rot)
            else:
                raise ValueError("Rotation matrix must have shape [..., 3, 3] or [..., 2, 3] or [..., 6]")
        else: # axis-angle
            assert self._rot.shape[-1] == 3, "Axis-angle must have shape [..., 3]"

    def apply_rotation(self, rot):
        """
        Apply rotation to the current rotation, left-multiplication.
        R_new = R_incoming ⋅ R_current
        """
        assert rot.rot_type in ["quat", "rmat"]

        rot = rot.convert(self._rot_type)
        if self._rot_type == "quat":
            new_rot = quaternion_multiply(rot._rot, self._rot)
        else:
            new_rot = torch.matmul(rot._rot, self._rot)

        return self.__class__(new_rot, rot_type=self._rot_type)

    def convert(self, rot_type):
        """
        Convert rotation to a different representation.
        """
        assert rot_type in self.ROT_TYPE, f"Rotation {rot_type} not supported."

        src_type = self.ROT_NAME[self._rot_type]
        dst_type = self.ROT_NAME[rot_type]

        if src_type == dst_type:
            return self.clone()
        
        new_rot = eval(f"{src_type}_to_{dst_type}")(self._rot)
        return self.__class__(new_rot, rot_type=rot_type)

    def to_quat(self):
        """
        Convert rotation to quaternion representation.
        """
        return self.convert("quat").rot

    def to_rmat(self):
        """
        Convert rotation to rotation matrix representation.
        """
        return self.convert("rmat").rot

    def to_axis_angle(self):
        """
        Convert rotation to axis-angle representation.
        """
        return self.convert("axis").rot

    def to_euler(self, order="zyx", to_degree=True):
        """
        Convert rotation to Euler angles.
        """
        quat = self.to_quat()
        assert quat.shape[-1] == 4, "Quaternion must have shape [..., 4]"

        original_shape = list(quat.shape)
        original_shape[-1] = 3  # Euler angles have 3 components
        quat = quat.view(-1, 4)  # flatten to 2D for processing

        q0 = quat[:, 0]
        q1 = quat[:, 1]
        q2 = quat[:, 2]
        q3 = quat[:, 3]

        if order == "xyz":
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 *q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == "yzx":
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == "zxy":
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == "xzy":
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == "yxz":
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == "zyx":
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise

        euler = torch.stack((x, y, z), dim=1).view(original_shape)
        if to_degree:
            euler = euler * 180.0 / np.pi
        return euler

    @staticmethod
    def cat(rot_lst, dim=0):
        """Concat a list a Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.cat(rot_lst, dim=dim), rot_type)

    @staticmethod
    def stack(rot_lst, dim=0):
        """Stack a list of Rotation3D object."""
        assert isinstance(rot_lst, (list, tuple))
        assert all([isinstance(rot, Rotation3D) for rot in rot_lst])
        rot_type = rot_lst[0].rot_type
        assert all([rot.rot_type == rot_type for rot in rot_lst])
        rot_lst = [rot.rot for rot in rot_lst]
        return Rotation3D(torch.stack(rot_lst, dim=dim), rot_type)


    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot):
        self._rot = rot
        self._check_valid()

    @property
    def rot_type(self):
        return self._rot_type

    @property
    def shape(self):
        return self._rot.shape

    def reshape(self, *shape):
        return self.__class__(self._rot.reshape(*shape), self._rot_type)

    def view(self, *shape):
        return self.__class__(self._rot.view(*shape), self._rot_type)

    def squeeze(self, dim=None):
        return self.__class__(self._rot.squeeze(dim), self._rot_type)

    def unsqueeze(self, dim=None):
        return self.__class__(self._rot.unsqueeze(dim), self._rot_type)

    def flatten(self, *args, **kwargs):
        return self.__class__(self._rot.flatten(*args, **kwargs), self._rot_type)

    def unflatten(self, *args, **kwargs):
        return self.__class__(self._rot.unflatten(*args, **kwargs), self._rot_type)

    def transpose(self, *args, **kwargs):
        return self.__class__(self._rot.transpose(*args, **kwargs), self._rot_type)

    def permute(self, *args, **kwargs):
        return self.__class__(self._rot.permute(*args, **kwargs), self._rot_type)

    def contiguous(self):
        return self.__class__(self._rot.contiguous(), self._rot_type)

    def __getitem__(self, key):
        return self.__class__(self._rot[key], self._rot_type)

    def __len__(self):
        return self._rot.shape[0]

    @property
    def device(self):
        return self._rot.device

    def to(self, device):
        return self.__class__(self._rot.to(device), self._rot_type)

    def cuda(self, device=None):
        return self.__class__(self._rot.cuda(device), self._rot_type)

    @property
    def dtype(self):
        return self._rot.dtype

    def type(self, dtype):
        return self.__class__(self._rot.type(dtype), self._rot_type)

    def type_as(self, other):
        return self.__class__(self._rot.type_as(other), self._rot_type)

    def detach(self):
        return self.__class__(self._rot.detach(), self._rot_type)

    def clone(self):
        return self.__class__(self._rot.clone(), self._rot_type)
