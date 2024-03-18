import uuid
from abc import ABC, abstractmethod
from copy import copy

import numpy as np


class Parameter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())
        super().__init__()

    def __float__(self):
        return self.value

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class FloatParameter(Parameter):

    def __init__(self, name: str, value=0., min=-1., max=1., power=1.):
        super().__init__(name=name)
        self._value = value
        self.initial = copy(value)
        self.min = min
        self.max = max
        self.power = power

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def reset(self):
        self._value = self.initial


class IntParameter(FloatParameter):
    def __init__(self, name: str, value: int = 0, min: int = 0, max: int = 100):
        super().__init__(name=name, value=value, min=min, max=max)


class Float3Parameter(FloatParameter):
    @property
    def value(self):
        return np.array(self._value)

    @value.setter
    def value(self, value):
        self._value = np.array(value)


class Transform(Parameter):
    def __init__(self, translation=None, scale=None):
        super().__init__(name='transform')
        if translation is None:
            translation = Float3Parameter('translation', np.zeros((3,)))
        if scale is None:
            scale = FloatParameter('scale', 1, 0.01, 2)
        self.translation = translation
        self.scale = scale

    def __call__(self, points: np.ndarray, inv=True) -> np.ndarray:
        if inv:
            return (points - self.translation.value) / self.scale.value
        else:
            return points * self.scale.value + self.translation.value

    @property
    def value(self):
        raise NotImplementedError()

    def reset(self):
        self.translation.reset()
        self.scale.reset()
