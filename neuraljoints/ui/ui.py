import time

import numpy as np
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.base import Entity
from neuraljoints.ui.drawable import Drawable
from neuraljoints.ui.io import IOHandler
from neuraljoints.ui.wrappers.base_wrapper import get_wrapper
from neuraljoints.ui.wrappers.grid import IMPLICIT_PLANE
from neuraljoints.utils.utils import redirect_stdout


class FPSCounter:
    BUFFER = 30

    def __init__(self):
        self.prev_time = time.time_ns()
        self.fps_list = [0.]

    @property
    def current_fps(self):
        return self.fps_list[-1]

    @property
    def smooth_fps(self):
        return sum(self.fps_list) / len(self.fps_list)

    @property
    def min_fps(self):
        return min(self.fps_list)

    def __str__(self):
        return f'Fps: {self.current_fps:.1f}/{self.smooth_fps:.0f}/{self.min_fps:.0f}'

    def update(self):
        nanoseconds = time.time_ns()
        fps = 1e9 / (nanoseconds - self.prev_time)
        self.prev_time = nanoseconds
        self.fps_list.append(fps)
        self.fps_list = self.fps_list[-FPSCounter.BUFFER:]


class UIHandler:
    drawables: list[Drawable] = [IMPLICIT_PLANE]
    show_origo = False
    stdout = redirect_stdout()
    fps_counter = FPSCounter()
    open = False

    @classmethod
    def init(cls):
        ps.set_program_name('Neural Joints')
        ps.init()
        ps.set_automatically_compute_scene_extents(False)
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_build_default_gui_panels(False)
        ps.set_ground_plane_mode('none')
        ps.load_color_map('blue-red', 'media/colormap.png')
        io = imgui.GetIO()
        UIHandler.font = io.Fonts.AddFontFromFileTTF("media/IBMPlexMono-Regular.ttf", 20.)
        UIHandler.__set_camera()
        UIHandler.open = True

        ps.set_user_callback(lambda: UIHandler.update())

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

        cls.fps_counter.update()
        imgui.Text(str(cls.fps_counter))

        if imgui.TreeNode('Output'):
            imgui.BeginChild('Output', (0, 150), True)
            output = cls.stdout.getvalue()
            imgui.Text(output)
            imgui.EndChild()
            imgui.TreePop()

        for drawable in cls.drawables:
            drawable.draw(refresh=cls.drawables[0].changed)
        imgui.End()

        if ps.window_requests_close():
            UIHandler.open = False

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

