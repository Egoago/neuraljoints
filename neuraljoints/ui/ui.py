import time

import numpy as np
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.base import Entity
from neuraljoints.ui.drawable import Drawable
from neuraljoints.ui.io import IOHandler
from neuraljoints.ui.wrappers.base_wrapper import get_wrapper
from neuraljoints.ui.wrappers.implicit_wrapper import IMPLICIT_PLANE
from neuraljoints.utils.utils import redirect_stdout


class UIHandler:
    drawables: list[Drawable] = [IMPLICIT_PLANE]
    show_origo = False
    stdout = redirect_stdout()
    last_draw = None

    @classmethod
    def init(cls):
        ps.init()
        ps.set_automatically_compute_scene_extents(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_build_default_gui_panels(False)
        # ps.set_navigation_style("planar")
        # ps_plane = ps.add_scene_slice_plane()
        # ps_plane.set_pose((0, 0, 0.05), (0, 0, -1))
        # ps_plane.set_draw_plane(True)
        # ps_plane.set_draw_widget(True)
        ps.set_ground_plane_mode('none')
        ps.load_color_map('blue-red', 'media/colormap.png')
        io = imgui.GetIO()
        cls.font = io.Fonts.AddFontFromFileTTF("media/IBMPlexMono-Regular.ttf", 25.)
        cls.__add_base_vectors()
        cls.__set_camera()

    @classmethod
    def add_entities(cls, entities: list[Entity]):
        for entity in entities:
            cls.add_entity(entity)

    @classmethod
    def add_entity(cls, entity: Entity):
        cls.drawables.append(get_wrapper(entity))

    @classmethod
    def update(cls):
        imgui.PushFont(cls.font)
        imgui.Begin("NeuralJoints", True, imgui.ImGuiWindowFlags_AlwaysAutoResize)
        io = imgui.GetIO()
        IOHandler.update(io)

        if imgui.SmallButton('Reset camera'):
            cls.__set_camera()
        imgui.SameLine()
        cls.show_origo = imgui.Checkbox('Show origo', cls.show_origo)[1]
        cls.__add_base_vectors()

        imgui.SameLine()
        cls.__draw_fps()

        if imgui.TreeNode('Output'):
            imgui.BeginChild('Output', (0, 150), True)
            output = cls.stdout.getvalue()
            imgui.Text(output)
            imgui.EndChild()
            imgui.TreePop()

        for drawable in cls.drawables:
            drawable.draw(refresh=cls.drawables[0].changed)
        imgui.End()

    @classmethod
    def __set_camera(cls):
        intrinsics = ps.CameraIntrinsics(fov_vertical_deg=60., aspect=1.)
        extrinsics = ps.CameraExtrinsics(root=(0., 0., 4.), look_dir=(0, 0., -1.), up_dir=(0., 1., 0.))
        params = ps.CameraParameters(intrinsics, extrinsics)

        # set the viewport view to those parameters
        ps.set_view_camera_parameters(params)

    @classmethod
    def __add_base_vectors(cls):
        if cls.show_origo:
            pc = ps.register_point_cloud("origo", np.zeros((1, 3)))
            kwargs = {'vectortype': 'ambient', 'enabled': True, 'radius': 0.05}
            vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for vector, axis in zip(vectors, ['x', 'y', 'z']):
                pc.add_vector_quantity(axis, np.array([vector]), color=vector, **kwargs)
        else:
            ps.remove_point_cloud("origo", False)

    @classmethod
    def __draw_fps(cls):
        nanoseconds = time.time_ns()
        if cls.last_draw is not None:
            imgui.Text(f'{1e9 / (nanoseconds - cls.last_draw):.1f}fps')
        cls.last_draw = nanoseconds
