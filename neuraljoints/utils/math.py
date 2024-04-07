import numpy as np


def normalize(array: np.ndarray, axis=-1, order=2) -> np.ndarray:
    norm = np.linalg.norm(array, order, axis)
    return array / np.expand_dims(norm, axis)


def euler_to_rotation_matrix(rpy: np.ndarray, order='zyx'):
    """
    Convert roll, pitch, yaw angles to a rotation matrix.

    Parameters:
        roll: float, angle in radians around the x-axis.
        pitch: float, angle in radians around the y-axis.
        yaw: float, angle in radians around the z-axis.

    Returns:
        R: numpy array, 3x3 rotation matrix.
    """
    c_roll, c_pitch, c_yaw = np.cos(rpy).tolist()
    s_roll, s_pitch, s_yaw = np.sin(rpy).tolist()

    rotations = {'x': np.array([[1, 0, 0],
                                [0, c_roll, -s_roll],
                                [0, s_roll, c_roll]]),
                 'y': np.array([[c_pitch, 0, s_pitch],
                                [0, 1, 0],
                                [-s_pitch, 0, c_pitch]]),
                 'z': np.array([[c_yaw, -s_yaw, 0],
                                [s_yaw, c_yaw, 0],
                                [0, 0, 1]])}
    R = np.eye(3, dtype=rpy.dtype)
    for axis in order.lower():
        R = R @ rotations[axis]
    return R
