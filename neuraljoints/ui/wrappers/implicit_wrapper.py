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
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter, Float3Parameter, ChoiceParameter


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
        self.render = ChoiceParameter('render', 'normal', ['plain', 'normal', 'depth', 'position'])
        self._mesh = None
        self._points = None
        self._grid = None
        self.selected: ImplicitWrapper = None

    @torch.no_grad()
    def add_scalar_texture(self, name: str, implicit: Implicit = None, values: Union[np.ndarray, torch.Tensor] = None):
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
                                      enabled=True, **scalar_args)

    @torch.no_grad()
    def add_vector_field(self, name: str, implicit: Implicit = None, values: Union[np.ndarray, torch.Tensor] = None):
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
                                      enabled=True)

    @torch.no_grad()
    def get_points(self, device='cpu'):
        if self._points is None:
            bounds = self.bounds.value
            res = self.resolution.value
            x, y = torch.meshgrid(torch.linspace(-bounds[0], bounds[0], res, device=device),
                                  torch.linspace(bounds[1], -bounds[1], res, device=device),
                                  indexing="xy")
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

    def select(self, implicit_wrapper: 'ImplicitWrapper'):
        if self.selected is not None:
            if implicit_wrapper is None or implicit_wrapper.implicit.name != self.selected.implicit.name:
                self.mesh.remove_quantity(self.selected.implicit.name)
                self.grid.remove_quantity(self.selected.implicit.name)
                ps.remove_surface_mesh(self.selected.implicit.name, False)
        self.selected = implicit_wrapper

    def remove(self, implicit_wrapper: 'ImplicitWrapper'):
        if self.selected is not None and implicit_wrapper.implicit.name == self.selected.implicit.name:
            self.select(None)

    def draw_ui(self):
        super().draw_ui()
        if self.changed:
            self._mesh = None
            self._points = None
            self._grid = None
        if self.selected is not None:
            self.changed |= self.selected.changed
            if imgui.Button('trace'):
                self.sphere_trace()
            if ps.has_point_cloud('Sphere traced'):
                imgui.SameLine()
                if imgui.Button('remove'):
                    ps.remove_point_cloud('Sphere traced')
                    ps.request_redraw()

    def draw_geometry(self):
        super().draw_geometry()

        if self.selected is not None:
            implicit = self.selected.implicit
            x = self.get_points(implicit.device)
            x.requires_grad = self.gradient.value
            values = implicit(x)
            if self.gradient.value:
                grad = gradient(values, x)
                self.add_vector_field(implicit.name, values=grad)
            else:
                self.grid.remove_quantity(implicit.name)
            if self.surface.value:
                self.add_surface(implicit)
            else:
                ps.remove_surface_mesh(implicit.name, False)
            self.add_scalar_texture(implicit.name, values=values)

    def add_surface(self, implicit: Implicit):
        with torch.no_grad():
            bounds = self.bounds.value.to(implicit.device)
            res = self.resolution.value // 2
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

        if self.gradient.value or self.render.value == 'normal':
            vertices.requires_grad = True
            values = implicit(vertices)
            gradients = gradient(values, vertices)

        with torch.no_grad():
            if self.render.value == 'position':
                positions = (vertices/bounds[None, :] + 1.) / 2.
                sm.add_color_quantity('position', positions.detach().cpu().numpy(), enabled=True)
            elif self.render.value == 'depth':
                position = ps.get_view_camera_parameters().get_position()
                position = torch.tensor(position, device=vertices.device)
                depth = torch.linalg.norm(vertices-position[None, :], dim=-1)
                sm.add_scalar_quantity('depth', depth.detach().cpu().numpy(), enabled=True)
            elif self.render.value == 'normal':
                sm.add_color_quantity('Normals', ((gradients + 1.)/2.).detach().cpu().numpy(), enabled=True)

            if self.gradient.value:
                sm.add_vector_quantity('Normals', (gradients * 0.1).detach().cpu().numpy(),
                                       vectortype='ambient', defined_on='vertices',
                                       radius=0.01, color=(0.1, 0.1, 0.1), enabled=True)

    def get_cam_rays(self, device='cpu'):
        cam_params = ps.get_view_camera_parameters()

        corners = cam_params.generate_camera_ray_corners()
        corners = torch.tensor(corners, device=device)
        res = self.resolution.value * 8
        t = torch.linspace(0., 1., res, device=device, dtype=torch.float32)
        t_ = 1. - t
        top = t_[:, None] * corners[0][None, :] + t[:, None] * corners[1][None, :]
        bottom = t_[:, None] * corners[2][None, :] + t[:, None] * corners[3][None, :]
        directions = t_[:, None, None] * top[None, ...] + t[:, None, None] * bottom[None, ...]
        directions = torch.nn.functional.normalize(directions, dim=-1)

        position = cam_params.get_position()
        position = torch.tensor(position, device=device)

        return position, directions

    def project_rays_on_bbox(self, origo: torch.Tensor, directions: torch.Tensor, bounds: torch.Tensor):
        dirfrac = 1. / directions

        tlower = (-bounds - origo)[None, :] * dirfrac
        tupper = (bounds - origo)[None, :] * dirfrac

        tmin = torch.minimum(tlower, tupper).amax(dim=-1)
        tmax = torch.maximum(tlower, tupper).amin(dim=-1)

        mask = (tmax > 0) * (tmin < tmax)
        return tmin, mask

    def sphere_trace(self):
        if self.selected is not None:
            with torch.no_grad():
                iso = self.isosurface.value
                implicit = self.selected.implicit
                origo, directions = self.get_cam_rays(implicit.device)
                directions = directions.reshape(-1, 3)
                bounds = self.bounds.value.to(implicit.device)

                distance, bound_mask = self.project_rays_on_bbox(origo, directions, bounds)
                surface_mask = torch.zeros_like(bound_mask)
                trace_mask = ~surface_mask * bound_mask
                points = torch.ones_like(distance)[:, None] * origo

                points[trace_mask] = points[trace_mask] + distance[trace_mask, None] * directions[trace_mask]

                i = 0
                while trace_mask.sum() > 0 and i < 500:
                    i += 1
                    distance = implicit(points[trace_mask]) - iso
                    points[trace_mask] = points[trace_mask] + distance[..., None] * directions[trace_mask]

                    bound_mask[trace_mask] = torch.all(points[trace_mask].abs() < bounds[None, :], dim=-1)
                    surface_mask[trace_mask] = distance < 1e-3
                    trace_mask[trace_mask.clone()] = (~surface_mask[trace_mask]) * bound_mask[trace_mask]

            surface_mask = surface_mask * bound_mask
            pc = ps.register_point_cloud('Sphere traced', points[surface_mask].cpu().numpy(),
                                         point_render_mode='quad')
            #pc.add_scalar_quantity('mask', surface_mask.cpu().numpy(), enabled=True)

            if self.render.value == 'position':
                positions = (points[surface_mask]/bounds[None, :] + 1.) / 2.
                pc.add_color_quantity('position', positions.detach().cpu().numpy(), enabled=True)
            elif self.render.value == 'depth':
                depth = torch.linalg.norm(points[surface_mask]-origo[None, :], dim=-1)
                pc.add_scalar_quantity('depth', depth.detach().cpu().numpy(), enabled=True)
            elif self.render.value == 'normal':
                points = points[surface_mask]
                points.requires_grad = True
                distance = implicit(points)
                gradients = gradient(distance, points).detach().cpu().numpy()
                gradients = (gradients + 1.)/2.
                pc.add_color_quantity('gradients', gradients, enabled=True)


IMPLICIT_PLANE = ImplicitPlane()  # easier than using singleton


class ImplicitWrapper(EntityWrapper):
    TYPE = Implicit

    @property
    def implicit(self) -> Implicit:
        return self.object

    def draw_ui(self) -> bool:
        changed, select = imgui.Checkbox('', self == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self)
        elif self == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
        super().draw_ui()
        self.changed |= changed
        return self.changed

    def __del__(self):
        IMPLICIT_PLANE.remove(self)


class AggregateWrapper(SetWrapper, ImplicitWrapper):
    TYPE = Aggregate

    def draw_ui(self) -> bool:
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
