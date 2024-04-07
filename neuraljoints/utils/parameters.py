import uuid
from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from neuraljoints.utils.math import euler_to_rotation_matrix


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


class BoolParameter(Parameter):
    def __init__(self, name: str, value: bool):
        super().__init__(name)
        self._value = value
        self.initial = copy(value)

    def __bool__(self):
        return self._value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def reset(self):
        self._value = self.initial


class ChoiceParameter(Parameter):
    def __init__(self, name: str, value: str, choices: list[str]):
        super().__init__(name)
        self._value = value
        self.choices = choices
        self.initial = copy(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value in self.choices:
            self._value = value

    def reset(self):
        self._value = self.initial


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
    def __init__(self, translation=None, rotation=None, scale=None):
        super().__init__(name='transform')
        if translation is None:
            translation = Float3Parameter('translation', np.zeros((3,)))
        if rotation is None:
            rotation = Float3Parameter('rotation', np.zeros((3,)), min=-np.pi, max=np.pi)
        if scale is None:
            scale = FloatParameter('scale', 1, 0.01, 2)
        self.translation = translation
        self.rotation = rotation
        self.scale_param = scale

    def __call__(self, points: np.ndarray, inv=True) -> np.ndarray:
        if inv:
            return self.scale(self.rotate(self.translate(points,
                                                         inv=True),
                                              inv=True),
                                  inv=True)
        else:
            return self.translate(self.rotate(self.scale(points)))

    def scale(self, points: np.ndarray, inv=False):
        scale = self.scale_param.value
        if inv:
            scale = 1./scale
        return points * scale

    def translate(self, points: np.ndarray, inv=False):
        t = self.translation.value
        if inv:
            t = -t
        return points + t

    def rotate(self, points: np.ndarray, inv=False):
        R = euler_to_rotation_matrix(self.rotation.value)
        if inv:
            R = R.T
        return points @ R.T

    @property
    def value(self):
        raise NotImplementedError()

    def reset(self):
        self.translation.reset()
        self.rotation.reset()
        self.scale_param.reset()
