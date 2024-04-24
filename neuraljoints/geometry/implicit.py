import warnings
from abc import ABC, abstractmethod

import numpy as np

from neuraljoints.geometry.base import Entity, Proxy
from neuraljoints.utils.math import normalize
from neuraljoints.utils.parameters import Transform


class Implicit(Entity, ABC):
    def __init__(self, transform: Transform = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform if transform is not None else Transform()

    def __call__(self, position, grad=False):
        if self.transform is not None:
            position = self.transform(position)
        values = self.forward(position)
        if self.transform is not None:
            values = self.transform.scale(values)
        if grad:
            grads = self.gradient(position)
            if self.transform is not None:
                grads = self.transform.rotate(grads, inv=False)
            return values, grads
        return values

    def forward(self, position):
        warnings.warn(f'Forward not implemented for {self.__class__}')
        return np.zeros_like(position[..., 0])

    def gradient(self, position):
        warnings.warn(f'Gradient not implemented for {self.__class__}')
        return np.zeros_like(position)


class SDF(Implicit, ABC):
    pass


class Sphere(SDF):
    def forward(self, position):
        return np.linalg.norm(position, axis=-1) - 1

    def gradient(self, position):
        return normalize(position)


class Cube(SDF):
    def forward(self, position):
        d = abs(position) - 1
        return (np.linalg.norm(np.maximum(d, 0), axis=-1) +
                np.minimum(np.max(d, axis=-1), 0))

    def gradient(self, position):
        d = abs(position) - 1
        outer = normalize(np.maximum(d, 0))
        inner = np.eye(position.shape[-1])[np.argmax(d, axis=-1)]
        return np.where((np.max(d, axis=-1) > 0)[..., None], outer, inner) * np.sign(position)


class Cylinder(SDF):
    def forward(self, position):
        position[..., 1] = 0
        return np.linalg.norm(position, axis=-1) - 1

    def gradient(self, position):
        position[..., 1] = 0
        return normalize(position)


class Plane(SDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, position):
        return position[..., 1]

    def gradient(self, position):
        grad = np.zeros_like(position)
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
        return np.zeros_like(position[..., 0])

    def gradient(self, position):
        if self.child is not None:
            return -self.child(position, grad=True)[1]
        return np.zeros_like(position)


class SdfToUdf(ImplicitProxy):
    @property
    def sdf(self) -> SDF:
        return self.child

    def forward(self, position):
        if self.child is not None:
            return np.abs(self.child(position))
        return np.zeros_like(position[..., 0])

    def gradient(self, position):
        if self.child is not None:
            sdf = self.sdf(position)
            return self.sdf(position, grad=True)[1] * np.sign(sdf)[..., None]
        return np.zeros_like(position)

