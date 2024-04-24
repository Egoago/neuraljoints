from abc import abstractmethod

import numpy as np

from neuraljoints.geometry.base import Set
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.utils.parameters import FloatParameter


class Aggregate(Implicit, Set):
    def forward(self, position):
        if len(self.children) == 0:
            return np.zeros_like(position[..., 0])
        return self.reduce(self.foreach(lambda child: child(position)))

    def gradient(self, position):
        if len(self.children) == 0:
            return np.zeros_like(position)
        else:
            values, gradients = list(zip(*self.foreach(lambda child: child(position, grad=True))))
            return self.reduce_gradient(values, gradients)

    @abstractmethod
    def reduce(self, values: [np.ndarray]):
        pass

    @abstractmethod
    def reduce_gradient(self, values: [np.ndarray], gradients: [np.ndarray]):
        pass


class Union(Aggregate):
    def reduce(self, values: [np.ndarray]):
        return np.minimum.reduce(values)

    def reduce_gradient(self, values, gradients):
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-2)
        indices = np.argmin(values, axis=-1)
        return np.take_along_axis(gradients, indices[..., None, None], -2).squeeze()


class InverseUnion(Union):
    def reduce(self, values: [np.ndarray]):
        return -super().reduce(values)

    def reduce_gradient(self, values, gradients):
        return -super().reduce_gradient(values, gradients)


class Sum(Aggregate):
    def reduce(self, values: [np.ndarray]):
        return np.sum(values, axis=0)

    def reduce_gradient(self, values, gradients):
        gradients = np.stack(gradients, axis=-1)
        return gradients.sum(axis=-1)


class CleanUnion(Aggregate):
    def reduce(self, values: [np.ndarray]):
        values = np.stack(values, axis=-1)
        return values.sum(axis=-1) - np.sqrt((values**2).sum(axis=-1))

    def reduce_gradient(self, values, gradients):
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-1)
        return gradients.sum(axis=-1) - ((values[..., None, :] * gradients).sum(axis=-1)/np.sqrt((values**2).sum(axis=-1))[..., None])


class RUnion(Aggregate):
    def __init__(self, *args, **kwargs):
        self.a0 = FloatParameter('a0', 1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'a{i+1}'
            setattr(self, name, FloatParameter(name, 1, min=0., max=20.))

    def reduce(self, values: [np.ndarray]):
        values = np.stack(values, axis=-1)
        clean_union = values.sum(axis=-1) - np.sqrt((values**2).sum(axis=-1))
        a = np.array([getattr(self, f'a{i + 1}').value for i in range(len(self.children))])
        return clean_union + self.a0.value / (1 + ((values/a)**2).sum(axis=-1))

    def reduce_gradient(self, values: [np.ndarray], gradients: [np.ndarray]):
        raise NotImplementedError()


class RoundUnion(Union):
    def __init__(self, radius=1.0, *args, **kwargs):
        self.radius = FloatParameter('radius', radius, min=0., max=2.)
        super().__init__(*args, **kwargs)

    def reduce(self, values: [np.ndarray]):
        #https://www.ronja-tutorials.com/post/035-2d-sdf-combination/
        intersectionSpace = np.stack(values, axis=-1) - self.radius.value
        intersectionSpace = np.minimum(intersectionSpace, 0)
        insideDist = -np.linalg.norm(intersectionSpace, axis=-1)
        union = super().reduce(values)
        outsideDist = np.maximum(union, self.radius.value)
        return insideDist + outsideDist

    def reduce_gradient(self, values: [np.ndarray], gradients: [np.ndarray]):
        raise NotImplementedError()


class Intersect(Aggregate):
    def reduce(self, values: [np.ndarray]):
        return np.maximum.reduce(values)

    def reduce_gradient(self, values, gradients):
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-2)
        indices = np.argmax(values, axis=-1)
        return np.take_along_axis(gradients, indices[..., None, None], -2).squeeze()


class SuperElliptic(Aggregate):
    def __init__(self, *args, **kwargs):
        self.t = FloatParameter('t', 1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'r{i+1}'
            setattr(self, name, FloatParameter(name, 1, min=0., max=20.))

    def reduce(self, values: [np.ndarray]):
        p = np.stack(values, axis=-1)
        r = np.array([getattr(self, f'r{i + 1}').value for i in range(len(self.children))])
        return 1 - (np.maximum((1 - p) / r, 0) ** self.t.value).sum(axis=-1)

    def reduce_gradient(self, values: [np.ndarray], gradients: [np.ndarray]):
        raise NotImplementedError()
