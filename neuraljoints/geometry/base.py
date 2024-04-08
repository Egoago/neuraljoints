from typing import List

import numpy as np

from neuraljoints.utils.parameters import Parameter


class Entity:
    __names = []

    def __init__(self, name=None):
        if name is None:
            _name = self.__class__.__name__
            count = 2
            while _name in Entity.__names:
                _name = f'{name} {count}'
                count += 1
            name = _name
        if name not in Entity.__names:
            Entity.__names.append(name)
        self.name = name
        super().__init__()

    @property
    def hparams(self):
        return [v for v in self.__dict__.values() if isinstance(v, Parameter)]

    @property
    def entities(self):
        entities = [v for v in self.__dict__.values() if isinstance(v, Entity)]
        # lists = [v for v in self.__dict__.values() if isinstance(v, List)]
        # for list in lists:
        #     if all(isinstance(item, Entity) for item in list):
        #         entities += list
        return entities


class ControlPoints(Entity):
    def __init__(self, count=1, dim=3, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.points = np.zeros((count, dim), dtype=np.float32)

    def __getitem__(self, i):
        return self.points[i]
