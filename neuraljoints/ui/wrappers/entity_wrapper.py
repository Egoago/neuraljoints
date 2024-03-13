from polyscope_bindings import imgui

from neuraljoints.geometry.base import Entity
from neuraljoints.ui.wrappers.parameter_wrapper import ParameterWrapper
from neuraljoints.ui.wrappers.wrapper import Wrapper


class EntityWrapper(Wrapper):
    def __init__(self, entity: Entity):
        super().__init__()
        self.entity = entity

    def draw_params(self) -> bool:
        changed = False
        for parameter in self.entity.parameters:
            changed = ParameterWrapper.draw(parameter) or changed
        return changed

    def draw(self) -> bool:
        if len(self.entity.parameters) > 0:
            imgui.Separator()
            imgui.Text(self.entity.name)
            return self.draw_params()
        return False
