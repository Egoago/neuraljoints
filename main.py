import numpy as np
import polyscope as ps

from neuraljoints.geometry.aggregate import RoundUnion, Union
from neuraljoints.geometry.implicit import Sphere, Cube, SDFToUDF, ParametricToImplicitBrute
from neuraljoints.geometry.parametric import CubicBezier
from neuraljoints.ui.ui import UIHandler


if __name__ == "__main__":
    UIHandler.init()

    implicits = [# RoundUnion(children=[Sphere(), Cube()]),
                 Union(children=[SDFToUDF(Sphere()),ParametricToImplicitBrute(CubicBezier())]),
                 #ParametricToImplicitBrute(CubicBezier()),
                ]

    #implicits[0].children[0].transform.translation._value = np.array([0.8, 0, 0])
    #implicits[0].children[1].transform.translation._value = np.array([-0.8, 0, 0])
    #value = 0
    UIHandler.add_entities(implicits)


    def draw_ui():
        global value
        # for implicit in implicits:
        #     imgui.BeginGroup()
        #
        #     values = ['value', 'fx', 'fy', 'fz']
        #     clicked, value = imgui.ListBox('value', 0, values)
        #
        #     def callback(p):
        #         return implicit(p, value=values[value])
        #
        #     if changed or clicked or startup:
        #         ps_grid.add_scalar_quantity_from_callable(implicit.name, callback,
        #                                                   isolines_enabled=value == 0,
        #                                                   **scalar_args)
        #     imgui.EndGroup()
        UIHandler.update()
    ps.set_user_callback(draw_ui)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.show()
