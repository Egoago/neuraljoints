from abc import ABC, abstractmethod

import numpy as np

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.parametric import Parametric
from neuraljoints.utils.parameters import FloatParameter, Transform, IntParameter


class Implicit(Entity, ABC):
    def __init__(self, transform: Transform = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform if transform is not None else Transform()

    def __call__(self, position, value='value'):
        if self.transform is not None:
            position = self.transform(position)
        if value == 'value':
            return self.forward(position)
        if value == 'fx':
            return self.gradient(position)[..., 0]
        if value == 'fy':
            return self.gradient(position)[..., 1]
        if value == 'fz':
            return self.gradient(position)[..., 2]

    @abstractmethod
    def forward(self, position):
        pass

    @abstractmethod
    def gradient(self, position):
        pass


class SDF(Implicit, ABC):
    pass


class Sphere(SDF):
    def __init__(self, radius=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = FloatParameter('radius', radius, min=0, max=2)

    def forward(self, position):
        return np.linalg.norm(position, axis=-1) - self.radius.value

    def gradient(self, position):
        return position / np.linalg.norm(position, axis=-1, keepdims=True)


class Cube(SDF):
    def __init__(self, size=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = FloatParameter('size', size, min=0.01, max=2.)

    def forward(self, position):
        d = abs(position) - self.size.value
        return (np.linalg.norm(np.maximum(d, 0), axis=-1) +
                np.minimum(np.max(d, axis=-1), 0))

    def gradient(self, position):
        d = abs(position) - self.size.value
        outer = np.maximum(d, 0)
        inner = np.minimum(d, 0)
        zeros = np.zeros_like(position)
        zeros[np.argmax(d, axis=-1)] = 1
        return np.where.norm(d > 0) + 1


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
        raise NotImplementedError()


class ParametricToImplicitBrute(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.resolution = IntParameter('resolution', value=10, min=3, max=200)

    def forward(self, position):
        parameters = np.linspace(0., 1., self.resolution.value, dtype=np.float32)
        points = self.parametric(parameters)
        return np.linalg.norm(position[:, None, :] - points[None, ...], axis=-1).min(axis=-1)

    def gradient(self, position):
        raise NotImplementedError()
