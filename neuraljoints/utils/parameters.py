import uuid
from abc import ABC
from copy import copy

import torch

from neuraljoints.utils.math import euler_to_rotation_matrix


class Parameter(ABC):
    def __init__(self, name: str, initial, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.initial = initial
        self._value = None
        self.value = copy(self.initial)
        self.changed = False
        self.id = str(uuid.uuid4())

    def __float__(self):
        return self.value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def reset(self, initial=None):
        if initial is not None:
            self.initial = initial
        self._value = self.initial


class BoolParameter(Parameter):
    def __bool__(self):
        return self.value


class ChoiceParameter(Parameter):
    def __init__(self, choices, **kwargs):
        self.choices = choices
        super().__init__(**kwargs)
        assert self.initial in self.choices

    @Parameter.value.setter
    def value(self, value):
        if value in self.choices:
            self._value = value

    def reset(self, initial=None):
        if initial in self.choices:
            super().reset(initial)


class FloatParameter(Parameter):
    def __init__(self, *args, min=-1., max=1., **kwargs):
        self.min = min
        self.max = max
        super().__init__(*args, **kwargs)

    @Parameter.value.setter
    def value(self, value):
        self._value = min(max(float(value), self.min), self.max)


class IntParameter(Parameter):
    def __init__(self, *args, min=-1, max=1, **kwargs):
        self.min = min
        self.max = max
        super().__init__(*args, **kwargs)

    @Parameter.value.setter
    def value(self, value):
        self._value = min(max(int(value), self.min), self.max)


class Float3Parameter(FloatParameter):
    @property
    def value(self):
        return torch.tensor(self._value, dtype=torch.float32)

    @value.setter
    def value(self, value):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)
        value = torch.clamp(value, self.min, self.max)
        self._value = value.tolist()


class Int3Parameter(IntParameter):
    @property
    def value(self):
        return torch.tensor(self._value, dtype=torch.int)

    @value.setter
    def value(self, value):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.int)
        value = torch.clamp(value, self.min, self.max)
        self._value = value.tolist()


class Transform(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name='transform', initial=None, **kwargs)
        self.translation = Float3Parameter(name='translation', initial=[0, 0, 0])
        self.rotation = Int3Parameter(name='rotation', initial=[0, 0, 0], min=-180, max=180)

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

    @value.setter
    def value(self, value):
        pass

    def reset(self, initial=None):
        self.translation.reset()
        self.rotation.reset()

