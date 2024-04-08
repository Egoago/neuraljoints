from typing import Callable

import numpy as np
import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter


class ImplicitWrapper(EntityWrapper):
    scalar_args = {'datatype': 'symmetric', 'cmap': 'blue-red',
                   'isolines_enabled': True, 'isoline_width': 0.05}
    RESOLUTION = IntParameter('resolution', 100, 2, 500)    #TODO add static params
    BOUND = FloatParameter('bound', 2, 1, 10)
    GRADIENT = BoolParameter('gradient', False)

    _mesh = None
    _grid = None

    def __init__(self, implicit: Implicit, **kwargs):
        super().__init__(entity=implicit, **kwargs)

    @classmethod
    def points(cls):
        linspace = np.linspace(-cls.BOUND.value, cls.BOUND.value, cls.RESOLUTION.value)
        x, y = np.meshgrid(linspace, -linspace)
        points = np.stack([x, y, np.zeros_like(x)], axis=-1)
        return points.reshape(-1, 3)

    @classmethod
    def add_scalar_texture(cls, name: str, func: Callable):
        texture = func(cls.points()).reshape(cls.RESOLUTION.value, cls.RESOLUTION.value)
        cls.mesh.add_scalar_quantity(name, texture, defined_on='texture', param_name="uv", **cls.scalar_args)

    @classmethod
    def add_color_texture(cls, name: str, func: Callable):
        texture = func(cls.points()).reshape(cls.RESOLUTION.value, cls.RESOLUTION.value, 3)
        texture = (texture + 1) / 2
        cls.mesh.add_color_quantity(name, texture, defined_on='texture', param_name="uv")

    @classmethod
    @property
    def mesh(cls):
        if ImplicitWrapper._mesh is None:
            vertices = np.array([[-1, 1, 0],
                                 [1, 1, 0],
                                 [1, -1, 0],
                                 [-1, -1, 0]]) * cls.BOUND.value

            faces = np.arange(4).reshape((1, 4))
            uv = np.array([[0, 1],
                           [1, 1],
                           [1, 0],
                           [0, 0]])

            ImplicitWrapper._mesh = ps.register_surface_mesh("Mesh", vertices, faces)
            ImplicitWrapper._mesh.add_parameterization_quantity("uv", uv, defined_on='vertices')
        return ImplicitWrapper._mesh

    @property
    def implicit(self) -> Implicit:
        return self.entity

    def draw_geometry(self):
        super().draw_geometry()
        if self.GRADIENT.value:
            ImplicitWrapper.add_color_texture(self.implicit.name, lambda p: self.implicit(p, grad=True)[-1])
        else:
            ImplicitWrapper.add_scalar_texture(self.implicit.name, self.implicit)


class AggregateWrapper(ImplicitWrapper):    #TODO move to drawable
    def __init__(self, implicit: Aggregate, children: [ImplicitWrapper], **kwargs):
        super().__init__(implicit, **kwargs)
        self.children = children

    def draw_ui(self):
        super().draw_ui()

        if imgui.TreeNode('children'):
            for child in self.children:
                child.draw_ui()
                self.changed = child.changed or self.changed
            imgui.TreePop()

    def draw_geometry(self):
        super().draw_geometry()
        for child in self.children:
            child.draw_geometry()
