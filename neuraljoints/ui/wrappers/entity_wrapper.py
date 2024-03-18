import uuid
from abc import ABC

from polyscope_bindings import imgui

from neuraljoints.geometry.base import Entity
from neuraljoints.ui.wrappers.parameter_wrapper import ParameterWrapper
from neuraljoints.ui.wrappers.wrapper import Wrapper


class EntityWrapper(Wrapper, ABC):
    def __init__(self, entity: Entity, entity_wrappers=None):
        super().__init__()
        self.entity = entity
        self.id = str(uuid.uuid4())
        self.entity_wrappers = entity_wrappers if entity_wrappers is not None else []

    def draw_entity_ui(self, entity: Entity):
        if len(entity.hparams) > 0:
            imgui.Separator()
            imgui.Text(entity.name)
            for parameter in entity.hparams:
                self.changed = ParameterWrapper.draw(parameter) or self.changed

    def draw_ui(self):
        imgui.PushId(self.id)
        super().draw_ui()
        hparams = self.entity.hparams
        if len(self.entity_wrappers) > 0 or len(hparams) > 0:
            imgui.Separator()
            imgui.Text(self.entity.name)
        for hparam in hparams:
            self.changed = ParameterWrapper.draw(hparam) or self.changed
        if len(self.entity_wrappers) > 0:
            if len(self.entity_wrappers) == 1:
                imgui.Indent(10)
                self.entity_wrappers[0].draw_ui()
                self.changed = self.entity_wrappers[0].changed or self.changed
                imgui.Unindent(10)
            elif imgui.TreeNode('entities'):
                for entity_wrapper in self.entity_wrappers:
                    entity_wrapper.draw_ui()
                    self.changed = entity_wrapper.changed or self.changed
                imgui.TreePop()
        imgui.PopID()

    def draw_geometry(self):
        super().draw_geometry()
        for entity_wrapper in self.entity_wrappers:
            entity_wrapper.draw_geometry()
