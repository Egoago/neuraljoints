from abc import ABC, abstractmethod

import numpy as np

from neuraljoints.geometry.base import Entity
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
        values = self.transform.scale(values)
        if grad:
            grads = self.gradient(position)
            grads = self.transform.rotate(grads, inv=False)
            return values, grads
        return values

    @abstractmethod
    def forward(self, position):
        pass

    @abstractmethod
    def gradient(self, position):
        pass


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


class ImplicitProxy(Implicit, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = None


class SDFToUDF(ImplicitProxy):
    def __init__(self, sdf: SDF, **kwargs):
        super().__init__(**kwargs)
        self.sdf = sdf

    def forward(self, position):
        return np.abs(self.sdf(position))

    def gradient(self, position):
        sdf = self.sdf(position)
        raise self.sdf(position, 'gradient') * np.sign(sdf)[..., None]
