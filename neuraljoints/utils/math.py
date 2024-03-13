import numpy as np


def normalize(array: np.ndarray, axis=-1, order=2) -> np.ndarray:
    norm = np.linalg.norm(array, order, axis)
    return array / np.expand_dims(norm, axis)
