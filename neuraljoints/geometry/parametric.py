from abc import abstractmethod, ABC

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.explicit import PointCloud
from neuraljoints.utils.parameters import Transform


class Parametric(Entity, ABC):
    CONTROL_POINT_COUNT = None

    def __init__(self, transform: Transform = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform if transform is not None else Transform()
        self.control_points = PointCloud(count=self.CONTROL_POINT_COUNT)

    def __call__(self, parameter):
        position = self.forward(parameter)
        return self.transform(position, inv=False)

    @abstractmethod
    def forward(self, parameter):
        pass

    @abstractmethod
    def gradient(self, parameter):
        pass

    @abstractmethod
    def gradgrad(self, parameter):
        pass


class CubicBezier(Parametric):
    CONTROL_POINT_COUNT = 4

    def matrix(self, t: torch.Tensor):
        return torch.tensor([[1, 0, 0, 0],
                             [-3, 3, 0, 0],
                             [3, -6, 3, 0],
                             [-1, 3, -3, 1]], dtype=t.dtype, device=t.device)

    def forward(self, t):
        t = torch.column_stack([torch.ones_like(t), t, t**2, t**3])
        return t @ self.matrix(t) @ self.control_points.points

    def gradient(self, t):
        t = torch.column_stack([torch.zeros_like(t), torch.ones_like(t), 2*t, 3 * (t ** 2)])
        return t @ self.matrix(t) @ self.control_points.points

    def gradgrad(self, t):
        t = torch.column_stack([torch.zeros_like(t), torch.zeros_like(t), 2 * torch.ones_like(t), 6 * t])
        return t @ self.matrix(t) @ self.control_points.points
