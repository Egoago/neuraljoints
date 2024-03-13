import numpy as np

from neuraljoints.utils.parameters import Parameter


class Entity:
    def __init__(self, name=None):
        self.name = name if name is not None else self.__class__.__name__
        super().__init__()

    @property
    def parameters(self):
        return [v for v in self.__dict__.values() if isinstance(v, Parameter)]


class ControlPoints(Entity):
    def __init__(self, count=1, dim=3, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.points = np.zeros((count, dim), dtype=np.float32)

    def __getitem__(self, i):
        return self.points[i]
