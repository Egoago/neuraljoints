import uuid
from copy import copy
from typing import List, Optional

import torch
from pydantic import BaseModel, Field, model_validator

from neuraljoints.utils.math import euler_to_rotation_matrix


class Parameter(BaseModel):
    name: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initial: float
    value_: float

    @model_validator(mode='before')
    @classmethod
    def pre_root(cls, values: dict) -> dict:
        if 'value_' not in values and 'initial' in values:
            values['value_'] = copy(values['initial'])
        return values

    def __float__(self):
        return self.value

    @property
    def value(self):
        return self.value_

    @value.setter
    def value(self, value):
        self.value_ = value

    def reset(self, initial=None):
        if initial is not None:
            self.initial = initial
        self.value_ = self.initial


class BoolParameter(Parameter):
    initial: bool
    value_: bool

    def __bool__(self):
        return self.value


class ChoiceParameter(Parameter):
    initial: str
    value_: str
    choices: List[str]

    @Parameter.value.setter
    def value(self, value):
        if value in self.choices:
            self.value_ = value


class FloatParameter(Parameter):
    min: float = -1.
    max: float = 1.

    @Parameter.value.setter
    def value(self, value):
        self.value_ = min(max(value, self.min), self.max)


class IntParameter(FloatParameter):
    initial: int
    value_: int
    min: int = -1
    max: int = 1


class Float3Parameter(FloatParameter):
    initial: List[float]
    value_: List[float]

    @property
    def value(self):
        return torch.tensor(self.value_, dtype=torch.float32)

    @value.setter
    def value(self, value):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)
        value = torch.clamp(value, self.min, self.max)
        self.value_ = value.tolist()


class Int3Parameter(IntParameter):
    initial: List[int]
    value_: List[int]

    @property
    def value(self):
        return torch.tensor(self.value_, dtype=torch.int)

    @value.setter
    def value(self, value):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.int)
        value = torch.clamp(value, self.min, self.max)
        self.value_ = value.tolist()


class Transform(Parameter):
    name: str = 'transform'
    translation: Float3Parameter = Float3Parameter(name='translation', initial=[0, 0, 0])
    rotation: Int3Parameter = Int3Parameter(name='rotation', initial=[0, 0, 0], min=-180, max=180)
    initial: Optional[float] = Field(exclude=True)
    value_: Optional[float] = Field(exclude=True)

    @model_validator(mode='before')
    @classmethod
    def pre_root(cls, values: dict) -> dict:
        values = super().pre_root(values)
        values['initial'] = None
        values['value_'] = None
        return values

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

