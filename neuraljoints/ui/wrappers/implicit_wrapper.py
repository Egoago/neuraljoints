from typing import Callable

import numpy as np
import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit, ImplicitProxy
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, ProxyWrapper
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter, Float3Parameter


class ImplicitPlane(EntityWrapper):
    TYPE = None

    def __init__(self):
        super().__init__(object=Entity(name='Grid'))
        self.resolution = IntParameter('resolution', 100, 2, 200)
        self.bounds = Float3Parameter('bounds', (2, 2, 2), 1, 10)
        self.z = FloatParameter('z', 0, -1, 1)
        self.gradient = BoolParameter('gradient', False)
        self._mesh = None
        self._points = None
        self._grid = None
        self.selected = None

    def add_scalar_texture(self, name: str, func: Callable = None, values: np.ndarray = None):
        if self.selected != name:
            return
        scalar_args = {'datatype': 'symmetric', 'cmap': 'blue-red',
                       'isolines_enabled': True, 'isoline_width': 0.1, 'isoline_darkness': 0.8}
        if values is None:
            if func is None:
                raise AttributeError('A callable function or the direct values have to provided.')
            values = func(self.points)
        texture = values.reshape(self.resolution.value, self.resolution.value)
        self.mesh.add_scalar_quantity(name, texture, defined_on='texture', param_name="uv",
                                      enabled=self.selected == name, **scalar_args)

    def add_vector_field(self, name: str, func: Callable = None, values: np.ndarray = None):
        if self.selected != name:
            return
        if values is None:
            if func is None:
                raise AttributeError('A callable function or the direct values have to provided.')
            points = self.points[::4, ::4].reshape(-1, 3)
            values = func(points)
        self.grid.add_vector_quantity(name, values, radius=0.01, length=0.1, color=(0.1, 0.1, 0.1),
                                      enabled=self.selected == name)

    @property
    def points(self):
        if self._points is None:
            bounds = self.bounds.value
            res = self.resolution.value
            x, y = np.meshgrid(np.linspace(-bounds[0], bounds[0], res),
                               np.linspace(bounds[1], -bounds[1], res))
            z = np.ones_like(x) * self.z.value * bounds[2]
            self._points = np.stack([x, y, z], axis=-1)
        return self._points

    @property
    def grid(self) -> ps.PointCloud:
        if self._grid is None:
            points = self.points[::4, ::4].reshape(-1, 3)

            self._grid = ps.register_point_cloud("Implicit grid", points, point_render_mode='quad')
            self._grid.set_radius(0)
        return self._grid

    @property
    def mesh(self) -> ps.SurfaceMesh:
        if self._mesh is None:
            z = self.z.value
            vertices = np.array([[-1, 1, z],
                                 [1, 1, z],
                                 [1, -1, z],
                                 [-1, -1, z]]) * self.bounds.value

            faces = np.arange(4).reshape((1, 4))
            uv = np.array([[0, 1],
                           [1, 1],
                           [1, 0],
                           [0, 0]])

            self._mesh = ps.register_surface_mesh("Implicit plane", vertices, faces)
            self._mesh.add_parameterization_quantity("uv", uv, defined_on='vertices')
        return self._mesh

    def select(self, name: str | None):
        if name != self.selected and self.selected is not None:
            self.mesh.remove_quantity(self.selected)
            self.grid.remove_quantity(self.selected)
        self.selected = name

    def draw_ui(self):
        super().draw_ui()
        if self.changed:
            self._mesh = None
            self._points = None
            self._grid = None


IMPLICIT_PLANE = ImplicitPlane()  # easier than using singleton


class ImplicitWrapper(EntityWrapper):
    TYPE = Implicit

    @property
    def implicit(self) -> Implicit:
        return self.object

    def draw_ui(self) -> bool:
        changed, select = imgui.Checkbox('', self.object.name == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self.object.name)
        elif self.object.name == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
        super().draw_ui()
        self.changed |= changed
        return self.changed

    def draw_geometry(self):
        super().draw_geometry()
        if IMPLICIT_PLANE.gradient.value:
            IMPLICIT_PLANE.add_vector_field(self.implicit.name, lambda p: self.implicit(p, grad=True)[-1])
        else:
            IMPLICIT_PLANE.grid.remove_quantity(self.implicit.name)
        IMPLICIT_PLANE.add_scalar_texture(self.implicit.name, self.implicit)

    def __del__(self):
        IMPLICIT_PLANE.mesh.remove_quantity(self.implicit.name)
        IMPLICIT_PLANE.grid.remove_quantity(self.implicit.name)


class AggregateWrapper(SetWrapper, ImplicitWrapper):
    TYPE = Aggregate

    def draw_ui(self) -> bool:
        changed, select = imgui.Checkbox('', self.object.name == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self.object.name)
        elif self.object.name == IMPLICIT_PLANE.selected:
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
        SetWrapper.draw_geometry(self)


class ImplicitProxyWrapper(ProxyWrapper, ImplicitWrapper):
    TYPE = ImplicitProxy

    def draw_ui(self) -> bool:  # TODO refactor
        changed, select = imgui.Checkbox('', self.object.name == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self.object.name)
        elif self.object.name == IMPLICIT_PLANE.selected:
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
