from typing import Union

import cumcubes
import numpy as np
import torch
import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit, ImplicitProxy
from neuraljoints.neural.autograd import gradient
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, ProxyWrapper
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter, Float3Parameter


class ImplicitPlane(EntityWrapper):
    TYPE = None

    def __init__(self):
        super().__init__(object=Entity(name='Grid'))
        self.resolution = IntParameter('resolution', 100, 2, 200)
        self.bounds = Float3Parameter('bounds', [2, 2, 2], 1, 10)
        self.z = FloatParameter('z', 0, -1, 1)
        self.gradient = BoolParameter('gradient', False)
        self.surface = BoolParameter('surface', False)
        self.smooth = BoolParameter('smooth', False)
        self.isosurface = FloatParameter('isosurface', 0, -1, 1)
        self._mesh = None
        self._points = None
        self._grid = None
        self.selected = None

    @torch.no_grad()
    def add_scalar_texture(self, name: str, implicit: Implicit = None, values: Union[np.ndarray, torch.Tensor] = None):
        if self.selected != name:
            return
        scalar_args = {'datatype': 'symmetric', 'cmap': 'blue-red',
                       'isolines_enabled': True, 'isoline_width': 0.1, 'isoline_darkness': 0.9}
        if values is None:
            if implicit is None:
                raise AttributeError('An implicit function or the direct values have to provided.')
            values = implicit(self.get_points(implicit.device)).detach().cpu().numpy()
        values = values.reshape(self.resolution.value, self.resolution.value)
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        self.mesh.add_scalar_quantity(name, values, defined_on='texture', param_name="uv",
                                      enabled=self.selected == name, **scalar_args)

    @torch.no_grad()
    def add_vector_field(self, name: str, implicit: Implicit = None, values: Union[np.ndarray, torch.Tensor] = None):
        if self.selected != name:
            return
        if values is None:
            if implicit is None:
                raise AttributeError('An implicit function or the direct values have to provided.')
            points = self.get_points(implicit.device)[::4, ::4].reshape(-1, 3)
            values = implicit(points)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if values.ndim != 2:
            if values.size // 3 != self.grid.n_points():    # TODO refactor
                values = values[::4, ::4]
            values = values.reshape(-1, 3)
        self.grid.add_vector_quantity(name, values, radius=0.01, length=0.1, color=(0.1, 0.1, 0.1),
                                      enabled=self.selected == name)

    @torch.no_grad()
    def get_points(self, device='cpu'):
        if self._points is None:
            bounds = self.bounds.value
            res = self.resolution.value
            x, y = torch.meshgrid(torch.linspace(-bounds[0], bounds[0], res, device=device),
                                  torch.linspace(bounds[1], -bounds[1], res, device=device))
            z = torch.ones_like(x) * self.z.value * bounds[2]
            self._points = torch.stack([x, y, z], dim=-1)
        return self._points

    @property
    def grid(self) -> ps.PointCloud:
        if self._grid is None:
            points = self.get_points()[::4, ::4].reshape(-1, 3).detach().cpu().numpy()

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
                                 [-1, -1, z]]) * self.bounds.value.numpy()

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

    def add_surface(self, implicit: Implicit):
        if self.selected != implicit.name:
            ps.remove_surface_mesh(implicit.name, False)
            return
        with torch.no_grad():
            bounds = torch.tensor(IMPLICIT_PLANE.bounds.value, device=implicit.device, dtype=torch.float32)
            res = IMPLICIT_PLANE.resolution.value // 2
            X, Y, Z = torch.meshgrid(torch.linspace(-bounds[0], bounds[0], res, device=implicit.device, dtype=torch.float32),
                                     torch.linspace(-bounds[1], bounds[1], res, device=implicit.device, dtype=torch.float32),
                                     torch.linspace(-bounds[2], bounds[2], res, device=implicit.device, dtype=torch.float32),
                                     indexing="ij")
            position = torch.stack([X, Y, Z], dim=-1)

            values = implicit(position)

            vertices, faces = cumcubes.marching_cubes(values, self.isosurface.value)
            vertices = vertices / res * 2 * bounds - bounds + bounds / res

        sm = ps.register_surface_mesh(implicit.name, vertices.cpu().numpy(), faces.cpu().numpy(),
                                      smooth_shade=self.smooth.value)

        if self.gradient.value:
            vertices.requires_grad = True
            values = implicit(vertices)
            gradients = gradient(values, vertices)

            sm.add_vector_quantity('Normals', (gradients * 0.1).detach().cpu().numpy(),
                                   vectortype='ambient', defined_on='vertices',
                                   radius=0.01, color=(0.1, 0.1, 0.1), enabled=True)


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

    def draw_geometry(self):    # TODO refactor
        super().draw_geometry()
        values = self.implicit(IMPLICIT_PLANE.get_points(self.implicit.device), grad=IMPLICIT_PLANE.gradient.value)
        if IMPLICIT_PLANE.gradient.value:
            values, grad = values
            IMPLICIT_PLANE.add_vector_field(self.implicit.name, values=grad)
        else:
            IMPLICIT_PLANE.grid.remove_quantity(self.implicit.name)
        if IMPLICIT_PLANE.surface.value:
            IMPLICIT_PLANE.add_surface(self.implicit)
        else:
            ps.remove_surface_mesh(self.implicit.name, False)
        IMPLICIT_PLANE.add_scalar_texture(self.implicit.name, values=values)

    def __del__(self):
        IMPLICIT_PLANE.mesh.remove_quantity(self.implicit.name)
        IMPLICIT_PLANE.grid.remove_quantity(self.implicit.name)
        ps.remove_surface_mesh(self.implicit.name, False)


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
