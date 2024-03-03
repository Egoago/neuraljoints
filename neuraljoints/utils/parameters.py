import uuid
from abc import ABC, abstractmethod

import numpy as np
from polyscope import imgui


class Parameter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def register_ui(self) -> bool:
        pass


class FloatParameter(Parameter):
    def __init__(self, name: str, value=0., min=0., max=1.):
        super().__init__(name=name)
        self.name = name
        self._value = value
        self.min = min
        self.max = max

    @property
    def value(self):
        return self._value

    def register_ui(self):
        imgui.PushID(self.id)
        changed, value = imgui.SliderFloat(self.name, self.value,
                                           v_min=self.min, v_max=self.max)
        imgui.PopID()
        if changed:
            self._value = value
        return changed


class Float3Parameter(FloatParameter):
    @property
    def value(self):
        return np.array(self._value)

    def register_ui(self):
        imgui.PushID(self.id)
        changed, value = imgui.SliderFloat3(self.name, self.value.tolist(),
                                            v_min=self.min, v_max=self.max)
        imgui.PopID()
        if changed:
            self._value = np.array(value)
        return changed
