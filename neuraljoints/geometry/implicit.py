from abc import ABC, abstractmethod

import numpy as np

from neuraljoints.utils.parameters import FloatParameter, Float3Parameter, Parameter


class Implicit(ABC):
    def __init__(self, name=None):
        self.name = name if name is not None else self.__class__.__name__
        self.translation = Float3Parameter(f'translation', np.zeros((3,)))

    def __call__(self, position):
        position = position - self.translation.value
        return self.forward(position)

    @abstractmethod
    def forward(self, position):
        pass

    def register_ui(self) -> bool:
        changed = False
        for attribute, value in self.__dict__.items():
            if isinstance(value, Parameter):
                changed = changed or value.register_ui()
        return changed


class SDF(Implicit, ABC):
    pass


class Sphere(SDF):
    def __init__(self, radius=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = FloatParameter('radius', radius)

    def forward(self, position) -> float:
        return np.sqrt(np.sum(position*position, -1)) - self.radius.value


class Cube(SDF):
    def __init__(self, size=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = FloatParameter('size', size)

    def forward(self, position):
        d = abs(position) - self.size.value
        return (np.linalg.norm(np.maximum(d, 0), axis=-1) +
                np.minimum(np.max(d, axis=-1), 0))


class Union(SDF):
    def __init__(self, children: list[SDF]):
        super().__init__()
        self.children = children

    def forward(self, position):
        return np.minimum.reduce(list(map(lambda c: c(position), self.children)))
