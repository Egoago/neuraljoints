import numpy as np
import polyscope as ps
from polyscope_bindings import imgui

from neuraljoints.geometry.base import ControlPoints
from neuraljoints.ui.io import IOListener
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.utils.math import normalize
from neuraljoints.utils.parameters import FloatParameter


class ControlPointsWrapper(IOListener, EntityWrapper):
    def __init__(self, control_points: ControlPoints, **kwargs):
        super().__init__(entity=control_points, **kwargs)
        self.initial = np.copy(control_points.points)
        self.entity.control_radius = FloatParameter('control point radius', value=0.04, min=0.03, max=0.1)
        self.selected = None
        self.moved = False

    @property
    def control_points(self) -> ControlPoints:
        return self.entity

    def draw_ui(self):
        super().draw_ui()
        self.changed = self.moved or self.changed
        imgui.Text("points")
        imgui.SameLine()
        if imgui.SmallButton('reset'):
            self.changed = True
            self.control_points.points = self.initial
        ps.register_point_cloud(self.control_points.name, self.control_points.points,
                                enabled=True, radius=self.entity.control_radius.value)

    def __select_at(self, screen_coords):
        self.selected = None
        ps.set_give_focus_on_show(True)
        world_pos = ps.screen_coords_to_world_position(screen_coords)
        lengths = np.linalg.norm(self.control_points.points - world_pos, axis=-1)
        min_index = np.argmin(lengths)
        if lengths[min_index] < 1.5 * self.entity.control_radius.value:
            self.selected = min_index

    def on_mouse_clicked(self, screen_coords, button):
        if button == imgui.ImGuiMouseButton_Left and imgui.GetIO().KeyCtrl:
            self.__select_at(screen_coords)

    def on_mouse_released(self, screen_coords, button):
        self.selected = None
        self.changed = False
        self.moved = False

    def on_mouse_down(self, screen_coords, button):
        self.moved = False
        if self.selected is not None:
            camera = ps.get_view_camera_parameters()
            cam_pos = camera.get_position()
            look_dir = normalize(camera.get_look_dir())
            ray_dir = normalize(ps.screen_coords_to_world_ray(screen_coords))
            point_pos = self.control_points.points[self.selected]

            cam_to_point = point_pos - cam_pos
            ray_proj_look = ray_dir @ look_dir
            cam_to_goal = ray_dir * (cam_to_point @ look_dir / ray_proj_look)
            goal = cam_pos + cam_to_goal

            self.control_points.points[self.selected] = goal
            self.moved = True
