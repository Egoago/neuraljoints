import uuid
from abc import ABC, abstractmethod
from copy import copy

import torch

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

    def reset(self, initial=None):
        if initial is not None:
            self.initial = initial
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
        self._value = min(max(value, self.min), self.max)

    def reset(self):
        self._value = self.initial


class IntParameter(FloatParameter):
    def __init__(self, name: str, value: int = 0, min: int = 0, max: int = 100):
        super().__init__(name=name, value=value, min=min, max=max)


class Float3Parameter(FloatParameter):
    def __init__(self, name: str, value: list[float], min: int = -1., max: int = 1.):
        assert isinstance(value, list)
        super().__init__(name=name, value=torch.tensor(value, dtype=torch.float32), min=min, max=max)

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @value.setter
    def value(self, value):
        self._value = torch.tensor(value, dtype=torch.float32)
        self._value = torch.clamp(self._value, self.min, self.max)


class Int3Parameter(FloatParameter):
    def __init__(self, name: str, value: list[int], min: int = 0, max: int = 100):
        assert isinstance(value, list)
        super().__init__(name=name, value=torch.tensor(value, dtype=torch.int), min=min, max=max)

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @value.setter
    def value(self, value):
        self._value = torch.tensor(value, dtype=torch.int)
        self._value = torch.clamp(self._value, self.min, self.max)


class Transform(Parameter):
    def __init__(self, translation=None, rotation=None):
        super().__init__(name='transform')
        if translation is None:
            translation = Float3Parameter('translation', [0, 0, 0])
        if rotation is None:
            rotation = Int3Parameter('rotation', [0, 0, 0], min=-180, max=180)
        self.translation = translation
        self.rotation = rotation

    def __call__(self, points: torch.Tensor, inv=True, vector=False) -> torch.Tensor:
        assert points.shape[-1] == 3
        R = euler_to_rotation_matrix(torch.deg2rad(self.rotation.value).to(points.device))
        t = self.translation.value.to(points.device)

        if vector:
            if inv:
                return points @ R
            return points @ R.T
        else:
            if inv:
                return (points - t) @ R
            return (points @ R.T) + t

    @property
    def matrix(self):
        T = torch.eye(4, dtype=torch.float32)
        R = euler_to_rotation_matrix(torch.deg2rad(self.rotation.value))
        T[:3, :3] = R
        T[:3, -1] = self.translation.value
        return T

    @property
    def matrix_inv(self):
        return torch.linalg.inv(self.matrix)

    def translate(self, points: torch.Tensor, inv=False):
        t = self.translation.value.to(points.device)
        if inv:
            t = -t
        return points + t

    def rotate(self, points: torch.Tensor, inv=False):
        R = euler_to_rotation_matrix(torch.deg2rad(self.rotation.value)).to(points.device)
        if inv:
            R = R.T
        return points @ R.T

    @property
    def value(self):
        return self.matrix

    def reset(self):
        self.translation.reset()
        self.rotation.reset()


def class_parameter(parameter: Parameter):
    def decorator(cls):
        setattr(cls, parameter.name.upper(), parameter)
        return cls
    return decorator
