import numpy as np
import polyscope as ps
import torch
from PIL import ImageFont, Image, ImageDraw

from neuraljoints.geometry.base import Entity
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper


class ColorBar(EntityWrapper):
    def __init__(self, z, bounds, offset=0.05):
        super().__init__(object=Entity(name='Color Bar'))
        self.font = ImageFont.truetype("media/IBMPlexMono-Regular.ttf", size=40., encoding="unic")
        self.max = None
        self.min = None
        self.offset = offset
        self.z = z
        self.bounds = bounds
        self.cmap = None
        self._mesh = None
        self.highlighted = None
        self.scalar_args = None
        self._text_meshes = {}

    def text_mesh(self, key: str, height=None):
        if key not in self._text_meshes or height is not None:
            if height is None:
                height = 1 if key == 'max' else -1
            z = self.z.value
            center = np.array([1.15, height, z]) * self.bounds.value.numpy()
            look_dir = np.array([0, 0, -0.1])
            center = center - look_dir

            intrinsics = ps.CameraIntrinsics(fov_vertical_deg=45., aspect=1.)
            extrinsics = ps.CameraExtrinsics(root=center, look_dir=look_dir, up_dir=(0., 1., 0.))
            params = ps.CameraParameters(intrinsics, extrinsics)

            mesh = ps.register_camera_view(f"{key} mesh", params, widget_focal_length=0.1, enabled=True)
            mesh.set_widget_thickness(0)
            self._text_meshes[key] = mesh
        return self._text_meshes[key]

    @property
    def mesh(self) -> ps.SurfaceMesh:
        if self._mesh is None:
            z = self.z.value
            vertices = np.array([[1.0 + self.offset, 1, z],
                                 [1.1 + self.offset, 1, z],
                                 [1.1 + self.offset, -1, z],
                                 [1.0 + self.offset, -1, z]]) * self.bounds.value.numpy()

            faces = np.arange(4).reshape((1, 4))
            uv = np.array([[0, 1],
                           [1, 1],
                           [1, 0],
                           [0, 0]])

            self._mesh = ps.register_surface_mesh(self.object.name, vertices, faces)
            self._mesh.add_parameterization_quantity("uv", uv, defined_on='vertices')
        return self._mesh

    def update(self, values: torch.Tensor, scalar_args, refresh=True):
        self.scalar_args = scalar_args
        if len(values) == 0:
            self.min = None
            self.max = None
        elif refresh or self.max is None:
            self.max = values.max().item()
            self.min = values.min().item()
        else:
            self.max = max(self.max, values.max().item())
            self.min = min(self.min, values.min().item())
        self.changed = True

    def reset(self):
        self.min = None
        self.max = None
        self.cmap = None
        self._mesh = None
        self.scalar_args = None
        self.highlighted = None
        for key in self._text_meshes:
            ps.remove_surface_mesh(f"{key} mesh", False)
        self._text_meshes = {}

    def draw_ui(self) -> bool:
        return super().draw_ui()

    def draw_geometry(self):
        if self.max is not None:
            values = np.linspace(self.max, self.min, 100)
            values = np.stack([values, values], axis=1)
            self.mesh.add_scalar_quantity(self.object.name, values, defined_on='texture', param_name="uv", enabled=True,
                                          **self.scalar_args)

            self.render_text(f"< {self.min:.3f}", 'min')
            self.render_text(f"< {self.max:.3f}", 'max')
            if self.highlighted is not None:
                height = (self.highlighted-self.min) / (self.max - self.min + 1e-7) * 2 - 1
                self.render_text(f"< {self.highlighted:.3f}", 'highlighted', height)
            else:
                ps.remove_surface_mesh(f"highlighted mesh", False)

        else:
            self.mesh.remove_quantity(self.object.name)
            self.text_mesh("max").remove_quantity("max text")
            self.text_mesh("min").remove_quantity("min text")

    def render_text(self, text, key, height=None):
        text_width = self.font.getlength(text)
        canvas = Image.new('RGBA', [int(text_width)*2+10, int(self.font.size)+20], color='#FFFFFF')
        draw = ImageDraw.Draw(canvas)
        draw.text([int(text_width)+10, 0], text, font=self.font, fill='#000000')
        data = np.asarray(canvas) / 255.0
        data[..., 3] = 1-data[..., 0]

        self.text_mesh(key, height).add_color_alpha_image_quantity(f"{key} text", data, enabled=True)
