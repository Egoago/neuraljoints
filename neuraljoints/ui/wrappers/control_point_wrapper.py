import numpy as np
import torch
import polyscope as ps
from polyscope_bindings import imgui

from neuraljoints.geometry.explicit import PointCloud
from neuraljoints.ui.io import IOListener
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper
from neuraljoints.utils.parameters import FloatParameter


class ControlPointsWrapper(IOListener, EntityWrapper):  # TODO fix torch <- numpy
    TYPE = PointCloud
    INSTANCES: list['ControlPointsWrapper'] = []

    def __init__(self, control_points: PointCloud, **kwargs):
        super().__init__(object=control_points, **kwargs)
        self.initial = np.copy(control_points.points)
        self.entity.control_radius = FloatParameter('control point radius', initial=0.04, min=0.03, max=0.1)
        self.selected = None
        self.moved = False
        ControlPointsWrapper.INSTANCES.append(self)

    @property
    def control_points(self) -> PointCloud:
        return self.entity

    def draw_ui(self):
        super().draw_ui()
        self.changed = self.moved or self.changed
        imgui.Text("points")
        imgui.SameLine()
        if imgui.Button('reset'):
            self.changed = True
            self.control_points.points = self.initial
        ps.register_point_cloud(self.control_points.name, self.control_points.points,
                                enabled=True, radius=self.entity.control_radius.value,
                                color=self.color)

    @classmethod
    def __select_at(cls, screen_coords):
        lengths = []
        ps.set_give_focus_on_show(True)
        world_pos = ps.screen_coords_to_world_position(screen_coords)
        for instance in cls.INSTANCES:
            instance.selected = None
            lengths.append(np.linalg.norm(instance.control_points.points - world_pos, axis=-1))
        lengths = np.array(lengths)
        inst_idx, cp_idx = np.unravel_index(np.argmin(lengths), lengths.shape)
        if lengths[inst_idx, cp_idx] < 1.5 * cls.INSTANCES[inst_idx].entity.control_radius.value:
            cls.INSTANCES[inst_idx].selected = cp_idx

    def on_mouse_clicked(self, screen_coords, button):
        if button == imgui.ImGuiMouseButton_Left and \
           imgui.GetIO().KeyCtrl and \
           self.selected is None:
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
