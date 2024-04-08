from abc import abstractmethod

import numpy as np
from polyscope import imgui

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.utils.parameters import FloatParameter


class Aggregate(Implicit):
    def __init__(self, children: list[Implicit], *args, **kwargs):
        self.children = children
        super().__init__(*args, **kwargs)

    def forward(self, position):
        return self.reduce(list(map(lambda c: c(position), self.children)))

    @abstractmethod
    def reduce(self, values: [np.ndarray]):
        pass

    def register_ui(self) -> bool:
        changed = super().register_ui()
        if imgui.CollapsingHeader('children'):
            for child in self.children:
                changed = child.register_ui() or changed

        return changed


class Union(Aggregate):
    def gradient(self, position):
        values, gradients = list(zip(*map(lambda c: c(position, grad=True), self.children)))
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-2)
        indices = np.argmin(values, axis=-1)
        return np.take_along_axis(gradients, indices[..., None, None], -2).squeeze()

    def reduce(self, values: [np.ndarray]):
        return np.minimum.reduce(values)


class Sum(Aggregate):
    def gradient(self, position):
        _, gradients = list(zip(*map(lambda c: c(position, grad=True), self.children)))
        gradients = np.stack(gradients, axis=-1)
        return gradients.sum(axis=-1)

    def reduce(self, values: [np.ndarray]):
        return np.sum(values, axis=0)


class CleanUnion(Aggregate):
    def gradient(self, position):
        values, gradients = list(zip(*map(lambda c: c(position, grad=True), self.children)))
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-1)
        return gradients.sum(axis=-1) - ((values[..., None, :] * gradients).sum(axis=-1)/np.sqrt((values**2).sum(axis=-1))[..., None])

    def reduce(self, values: [np.ndarray]):
        values = np.stack(values, axis=-1)
        return values.sum(axis=-1) - np.sqrt((values**2).sum(axis=-1))


class RUnion(Aggregate):
    def __init__(self, *args, **kwargs):
        self.a0 = FloatParameter('a0', 1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'a{i+1}'
            setattr(self, name, FloatParameter(name, 1, min=0., max=20.))

    def gradient(self, position):
        raise NotImplementedError()

    def reduce(self, values: [np.ndarray]):
        values = np.stack(values, axis=-1)
        clean_union = values.sum(axis=-1) - np.sqrt((values**2).sum(axis=-1))
        a = np.array([getattr(self, f'a{i + 1}').value for i in range(len(self.children))])
        return clean_union + self.a0.value / (1 + ((values/a)**2).sum(axis=-1))


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

    def gradient(self, position):
        raise NotImplementedError()


class Intersect(Aggregate):
    def gradient(self, position):
        values, gradients = list(zip(*map(lambda c: c(position, grad=True), self.children)))
        values = np.stack(values, axis=-1)
        gradients = np.stack(gradients, axis=-2)
        indices = np.argmax(values, axis=-1)
        return np.take_along_axis(gradients, indices[..., None, None], -2).squeeze()

    def reduce(self, values: [np.ndarray]):
        return np.maximum.reduce(values)


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

    def gradient(self, position):
        raise NotImplementedError()
