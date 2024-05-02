import torch


def homogeneous(x: torch.Tensor, vector=False):
    shape = x.shape
    h = torch.ones((*shape[:-1], shape[-1] + 1), dtype=x.dtype, device=x.device)
    if vector:
        h[..., -1] = 0
    h[..., :-1] = x
    return h


def homogeneous_inv(h, vector=False):
    if vector:
        return h[..., :-1]
    return h[..., :-1] / h[..., -1, None]


def euler_to_rotation_matrix(rpy: torch.Tensor, order='zyx'):
    """
    Convert roll, pitch, yaw angles to a rotation matrix.

    Parameters:
        roll: float, angle in radians around the x-axis.
        pitch: float, angle in radians around the y-axis.
        yaw: float, angle in radians around the z-axis.

    Returns:
        R: numpy array, 3x3 rotation matrix.
    """
    c_roll, c_pitch, c_yaw = torch.cos(rpy).tolist()
    s_roll, s_pitch, s_yaw = torch.sin(rpy).tolist()

    rotations = {'x': torch.tensor([[1, 0, 0],
                                    [0, c_roll, -s_roll],
                                    [0, s_roll, c_roll]],
                                   device=rpy.device, dtype=rpy.dtype),
                 'y': torch.tensor([[c_pitch, 0, s_pitch],
                                   [0, 1, 0],
                                   [-s_pitch, 0, c_pitch]],
                                   device=rpy.device, dtype=rpy.dtype),
                 'z': torch.tensor([[c_yaw, -s_yaw, 0],
                                    [s_yaw, c_yaw, 0],
                                    [0, 0, 1]],
                                   device=rpy.device, dtype=rpy.dtype)}
    R = torch.eye(3, device=rpy.device, dtype=rpy.dtype)
    for axis in order.lower():
        R = R @ rotations[axis]
    return R
