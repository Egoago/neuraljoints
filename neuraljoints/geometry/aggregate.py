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
        raise NotImplementedError()

    def reduce(self, values: [np.ndarray]):
        return np.minimum.reduce(values)


class ComplexUnion(Aggregate):
    def gradient(self, position):
        raise NotImplementedError()

    def reduce(self, values: [np.ndarray]):
        values = -np.column_stack(values)
        return values.max(axis=-1)


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


class Intersect(Aggregate):
    def reduce(self, values: [np.ndarray]):
        return np.maximum.reduce(values)
