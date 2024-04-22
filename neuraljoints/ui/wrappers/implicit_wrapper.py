from typing import Callable

import numpy as np
import polyscope as ps

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit, ImplicitProxy
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter, Float3Parameter, class_parameter


@class_parameter(IntParameter('resolution', 100, 2, 200))
@class_parameter(Float3Parameter('bounds', (2, 2, 2), 1, 10))
@class_parameter(FloatParameter('z', 0, -1, 1))
@class_parameter(BoolParameter('gradient', False))
class ImplicitWrapper(EntityWrapper):
    TYPE = Implicit

    _mesh = None
    _changed = False
    _points = None
    _grid = None

    @classmethod
    def add_scalar_texture(cls, name: str, func: Callable):
        scalar_args = {'datatype': 'symmetric', 'cmap': 'blue-red',
                       'isolines_enabled': True, 'isoline_width': 0.05}
        texture = func(cls.points).reshape(cls.RESOLUTION.value, cls.RESOLUTION.value)
        cls.mesh.add_scalar_quantity(name, texture, defined_on='texture', param_name="uv", **scalar_args)

    @classmethod
    def add_color_texture(cls, name: str, func: Callable):
        texture = func(cls.points).reshape(cls.RESOLUTION.value, cls.RESOLUTION.value, 3)
        texture = (texture + 1) / 2
        cls.mesh.add_color_quantity(name, texture, defined_on='texture', param_name="uv")

    @classmethod
    def add_vector_field(cls, name: str, func: Callable):
        points = cls.points[::4, ::4].reshape(-1, 3)
        vectors = func(points)
        cls.grid.add_vector_quantity(name, vectors, radius=0.01, length=0.1, color=(0.1, 0.1, 0.1))

    @classmethod
    def change_check(cls):
        if cls._changed:
            cls._mesh = None
            cls._changed = False
            cls._points = None
            cls._grid = None

    @classmethod
    @property
    def points(cls):
        cls.change_check()
        if cls._points is None:
            cls._changed = False
            bounds = cls.BOUNDS.value
            res = cls.RESOLUTION.value
            x, y = np.meshgrid(np.linspace(-bounds[0], bounds[0], res),
                               np.linspace(bounds[1], -bounds[1], res))
            z = np.ones_like(x) * cls.Z.value * bounds[2]
            cls._points = np.stack([x, y, z], axis=-1)
        return cls._points

    @classmethod
    @property
    def grid(cls) -> ps.PointCloud:
        cls.change_check()
        if ImplicitWrapper._grid is None:
            points = cls.points[::4, ::4].reshape(-1, 3)

            ImplicitWrapper._grid = ps.register_point_cloud("Implicit grid", points, point_render_mode='quad')
            ImplicitWrapper._grid.set_radius(0)
        return ImplicitWrapper._grid

    @classmethod
    @property
    def mesh(cls) -> ps.SurfaceMesh:
        cls.change_check()
        if ImplicitWrapper._mesh is None:
            z = cls.Z.value
            vertices = np.array([[-1, 1, z],
                                 [1, 1, z],
                                 [1, -1, z],
                                 [-1, -1, z]]) * cls.BOUNDS.value

            faces = np.arange(4).reshape((1, 4))
            uv = np.array([[0, 1],
                           [1, 1],
                           [1, 0],
                           [0, 0]])

            ImplicitWrapper._mesh = ps.register_surface_mesh("Implicit plane", vertices, faces)
            ImplicitWrapper._mesh.add_parameterization_quantity("uv", uv, defined_on='vertices')
        return ImplicitWrapper._mesh

    @property
    def implicit(self) -> Implicit:
        return self.object

    def draw_geometry(self):
        super().draw_geometry()
        if self.GRADIENT.value:
            ImplicitWrapper.add_vector_field(self.implicit.name + ' grad', lambda p: self.implicit(p, grad=True)[-1])
        else:
            ImplicitWrapper.add_scalar_texture(self.implicit.name, self.implicit)

    def __del__(self):
        self.mesh.remove_quantity(self.object.name)
        self.grid.remove_quantity(self.object.name + ' grad')


class AggregateWrapper(SetWrapper, ImplicitWrapper):
    TYPE = Aggregate

    @classmethod
    @property
    def choices(cls) -> set[Entity] | None:
        return {c for c in Implicit.subclasses if not issubclass(c, ImplicitProxy)}

    def draw_geometry(self):
        ImplicitWrapper.draw_geometry(self)
        SetWrapper.draw_geometry(self)
