from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit, ImplicitProxy
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, ProxyWrapper
from neuraljoints.ui.wrappers.grid import IMPLICIT_PLANE


class ImplicitWrapper(EntityWrapper):
    TYPE = Implicit

    def __init__(self, object: TYPE, **kwargs):
        super().__init__(object, **kwargs)
        self.scalar_args = {'datatype': 'symmetric', 'cmap': 'blue-red',
                            'isolines_enabled': True, 'isoline_width': 0.1, 'isoline_darkness': 0.9}

    @property
    def implicit(self) -> Implicit:
        return self.object

    @property
    def surface_implicit(self) -> Implicit:
        return self.implicit

    def draw_ui(self) -> bool:
        imgui.PushID(self.object.name)
        changed, select = imgui.Checkbox('', self == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self)
        elif self == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
        EntityWrapper.draw_ui(self)
        self.changed |= changed
        imgui.PopID()
        return self.changed

    def __del__(self):
        IMPLICIT_PLANE.remove(self)


class AggregateWrapper(SetWrapper, ImplicitWrapper):
    TYPE = Aggregate

    def draw_ui(self) -> bool:
        imgui.PushID(self.object.name)
        changed, select = imgui.Checkbox('', self == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self)
        elif self == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
        super().draw_ui()
        self.changed |= changed
        imgui.PopID()
        return self.changed

    @classmethod
    @property
    def choices(cls) -> set[Entity] | None:
        return Implicit.subclasses

    def draw_geometry(self):
        ImplicitWrapper.draw_geometry(self)
        SetWrapper.draw_geometry(self)


class ImplicitProxyWrapper(ProxyWrapper, ImplicitWrapper):
    TYPE = ImplicitProxy

    def draw_ui(self) -> bool:  # TODO refactor
        changed, select = imgui.Checkbox('', self == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self)
        elif self == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
        super().draw_ui()
        self.changed |= changed
        return self.changed

    @classmethod
    @property
    def choices(cls) -> set[Entity] | None:
        return Implicit.subclasses

    def draw_geometry(self):
        ImplicitWrapper.draw_geometry(self)
        ProxyWrapper.draw_geometry(self)
