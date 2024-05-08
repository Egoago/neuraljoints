import warnings
from abc import ABC

import torch

from neuraljoints.geometry.base import Entity, Proxy
from neuraljoints.utils.parameters import Transform, FloatParameter


class Implicit(Entity, ABC):
    def __init__(self, transform: Transform = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform if transform is not None else Transform()

    def __call__(self, position: torch.Tensor) -> torch.Tensor:
        if self.transform is not None:
            position = self.transform(position)
        return self.forward(position)

    def forward(self, position):
        warnings.warn(f'Forward not implemented for {self.__class__}')
        return torch.zeros_like(position[..., 0])


class SDF(Implicit, ABC):
    pass


class Sphere(SDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.radius = FloatParameter('radius', 1., 1e-5, 2.)

    def forward(self, position):
        return torch.linalg.norm(position, dim=-1) - self.radius.value


class Cube(SDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = FloatParameter('size', 2., 1e-5, 4.)

    def forward(self, position):
        d = abs(position) - self.size.value/2.
        return (torch.linalg.norm(torch.clamp(d, min=0), dim=-1) +
                torch.clamp(torch.amax(d, dim=-1), max=0))


class Cylinder(SDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.radius = FloatParameter('radius', 1., 1e-5, 2.)

    def forward(self, position):
        position[..., 1] = 0
        return torch.linalg.norm(position, dim=-1) - self.radius.value


class Plane(SDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, position):
        return position[..., 1]


class ImplicitProxy(Implicit, Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = None


class Inverse(ImplicitProxy):
    def forward(self, position):
        if self.child is not None:
            return -self.child(position)
        return torch.zeros_like(position[..., 0])


class SdfToUdf(ImplicitProxy):
    @property
    def sdf(self) -> SDF:
        return self.child

    def forward(self, position):
        if self.child is not None:
            return torch.abs(self.child(position))
        return torch.zeros_like(position[..., 0])
