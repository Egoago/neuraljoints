import warnings
from abc import ABC
from typing import Union, Tuple

import torch

from neuraljoints.geometry.base import Entity, Proxy
from neuraljoints.utils.parameters import Transform


class Implicit(Entity, ABC):
    def __init__(self, transform: Transform = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform if transform is not None else Transform()

    def __call__(self, position: torch.Tensor, grad=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.transform is not None:
            position = self.transform(position)
        values = self.forward(position)
        if grad:
            grads = self.gradient(position)
            if self.transform is not None:
               grads = self.transform.rotate(grads, inv=False)
            return values, grads
        return values

    def forward(self, position):
        warnings.warn(f'Forward not implemented for {self.__class__}')
        return torch.zeros_like(position[..., 0])

    def gradient(self, position):
        warnings.warn(f'Gradient not implemented for {self.__class__}')
        return torch.zeros_like(position)


class SDF(Implicit, ABC):
    pass


class Sphere(SDF):
    def forward(self, position):
        return torch.linalg.norm(position, dim=-1) - 1

    def gradient(self, position):
        return torch.nn.functional.normalize(position, dim=-1)


class Cube(SDF):
    def forward(self, position):
        d = abs(position) - 1
        return (torch.linalg.norm(torch.clamp(d, min=0), dim=-1) +
                torch.clamp(torch.amax(d, dim=-1), max=0))

    def gradient(self, position):
        d = abs(position) - 1
        outer = torch.nn.functional.normalize(torch.clamp(d, min=0), dim=-1)
        inner = torch.eye(position.shape[-1], device=position.device, dtype=position.dtype)[torch.argmax(d, dim=-1)]
        return torch.where((torch.amax(d, dim=-1) > 0)[..., None], outer, inner) * torch.sign(position)


class Cylinder(SDF):
    def forward(self, position):
        position[..., 1] = 0
        return torch.linalg.norm(position, dim=-1) - 1

    def gradient(self, position):
        position[..., 1] = 0
        return torch.nn.functional.normalize(position, dim=-1)


class Plane(SDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, position):
        return position[..., 1]

    def gradient(self, position):
        grad = torch.zeros_like(position)
        grad[..., 1] = 1
        return grad


class ImplicitProxy(Implicit, Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = None


class Inverse(ImplicitProxy):
    def forward(self, position):
        if self.child is not None:
            return -self.child(position)
        return torch.zeros_like(position[..., 0])

    def gradient(self, position):
        if self.child is not None:
            return -self.child(position, grad=True)[1]
        return torch.zeros_like(position)


class SdfToUdf(ImplicitProxy):
    @property
    def sdf(self) -> SDF:
        return self.child

    def forward(self, position):
        if self.child is not None:
            return torch.abs(self.child(position))
        return torch.zeros_like(position[..., 0])

    def gradient(self, position):
        if self.child is not None:
            sdf = self.sdf(position)
            return self.sdf(position, grad=True)[1] * torch.sign(sdf)[..., None]
        return torch.zeros_like(position)

