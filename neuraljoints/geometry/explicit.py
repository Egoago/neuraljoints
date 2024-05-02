import torch

from neuraljoints.geometry.base import Entity


class Explicit(Entity):
    pass


class PointCloud(Explicit, torch.nn.Module):
    def __init__(self, count=1, dim=3, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.points = torch.zeros((count, dim), dtype=torch.float32)    # TODO fix device

    def __getitem__(self, i):
        return self.points[i]
