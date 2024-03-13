import numpy as np
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit, ParametricToImplicitBrute
from neuraljoints.geometry.parametric import Parametric
from neuraljoints.ui.drawable import Drawable
from neuraljoints.ui.io import IOHandler
from neuraljoints.ui.wrappers.implicit_wrapper import ImplicitWrapper, AggregateWrapper, \
    ParametricToImplicitBruteWrapper
from neuraljoints.ui.wrappers.parametric_wrapper import ParametricWrapper


class UIHandler:
    drawables: list[Drawable] = []
    show_origo = False

    @classmethod
    def init(cls):
        ps.init()
        ps.set_automatically_compute_scene_extents(False)
        # ps.set_navigation_style("planar")
        # ps_plane = ps.add_scene_slice_plane()
        # ps_plane.set_pose((0, 0, 0.05), (0, 0, -1))
        # ps_plane.set_draw_plane(True)
        # ps_plane.set_draw_widget(True)
        ps.set_ground_plane_mode('none')
        ps.load_color_map('blue-red', 'colormap.png')
        io = imgui.GetIO()
        cls.font = io.Fonts.AddFontFromFileTTF("media/IBMPlexMono-Regular.ttf", 20.)
        cls.__add_base_vectors()
        cls.__set_camera()

    @classmethod
    def add_entities(cls, entities: list[Entity]):
        for entity in entities:
            cls.add_entity(entity)

    @classmethod
    def add_entity(cls, entity: Entity):    #TODO refactor
        if isinstance(entity, Aggregate):
            cls.drawables.append(AggregateWrapper(entity))
        elif isinstance(entity, ParametricToImplicitBrute):
            cls.drawables.append(ParametricToImplicitBruteWrapper(entity))
        elif isinstance(entity, Parametric):
            cls.drawables.append(ParametricWrapper(entity))
        elif isinstance(entity, Implicit):
            cls.drawables.append(ImplicitWrapper(entity))

    @classmethod
    def update(cls):
        io = imgui.GetIO()
        imgui.PushFont(cls.font)
        IOHandler.update(io)

        if imgui.SmallButton('Reset camera'):
            cls.__set_camera()
        imgui.SameLine()
        cls.show_origo = imgui.Checkbox('Show origo', cls.show_origo)[1]
        cls.__add_base_vectors()

        for drawable in cls.drawables:
            drawable.draw()

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
