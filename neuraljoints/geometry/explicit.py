import numpy as np

from neuraljoints.geometry.base import Entity


class Explicit(Entity):
    pass


class PointCloud(Explicit):
    def __init__(self, count=1, dim=3, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.points = np.zeros((count, dim), dtype=np.float32)

    def __getitem__(self, i):
        return self.points[i]
