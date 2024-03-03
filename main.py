import numpy as np
import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.implicit import Sphere, Cube, Union


def add_base_vectors():
    # Define coordinate basis vectors
    nodes = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    edges = np.array([[0, 1], [0, 2], [0, 3]])

    # Register basis vectors as line segments
    ps_net = ps.register_curve_network("Base", nodes, edges)
    colors = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    ps_net.add_color_quantity("color", colors, enabled=True, defined_on='edges')
    ps_net.set_ignore_slice_plane(ps_plane, True)


if __name__ == "__main__":
    ps.init()
    #ps.set_navigation_style("planar")
    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_pose((0, 0, 0.05), (0, 0, -1))
    #ps_plane.set_draw_plane(True)
    #ps_plane.set_draw_widget(True)
    add_base_vectors()

    # define the resolution and bounds of the grid
    cube = Cube()
    dims = (100, 100, 100)
    bound_low = (-2, -2, -2)
    bound_high = (2, 2, 2)

    # register the grid
    ps_grid = ps.register_volume_grid("Grid", dims, bound_low, bound_high)
    ps_grid.set_cull_whole_elements(False)

    implicits = [Cube(), Sphere()]
    implicits.append(Union(implicits.copy()))

    def draw_ui():
        scalar_args = {'datatype': 'symmetric', 'isolines_enabled': True,
                       'enable_isosurface_viz': True, 'isosurface_color': (0.4, 0.6, 0.6)}
        for implicit in implicits:
            imgui.Text(implicit.name)
            changed = implicit.register_ui()
            imgui.Separator()

            if changed:
                ps_grid.add_scalar_quantity_from_callable(implicit.name, implicit, **scalar_args)

    ps.set_user_callback(draw_ui)
    ps.show()
