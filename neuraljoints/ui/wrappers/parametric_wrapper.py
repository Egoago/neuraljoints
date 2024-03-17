import numpy as np
import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.parametric import Parametric
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.utils.math import normalize


class ParametricWrapper(EntityWrapper):
    RESOLUTION = 100

    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(entity=parametric, **kwargs)
        self.show_curvature_comb = False

    @property
    def parametric(self) -> Parametric:
        return self.entity

    def draw_ui(self):
        super().draw_ui()
        checkbox_result = imgui.Checkbox('Show curvature comb', self.show_curvature_comb)
        self.changed = checkbox_result[0] != checkbox_result[1] or self.changed
        self.show_curvature_comb = checkbox_result[1]

    def draw_geometry(self):
        super().draw_geometry()
        count = self.RESOLUTION
        parameters = np.linspace(0, 1, count, dtype=np.float32)
        points = self.parametric(parameters)
        tangent = self.parametric.gradient(parameters)
        acceleration = self.parametric.gradgrad(parameters)

        binormal = np.cross(tangent, acceleration)
        normal = np.cross(binormal, tangent)

        cn = ps.register_curve_network(self.parametric.name, points, 'line')
        cn.add_vector_quantity('vector', points, vectortype='ambient', radius=0.01)
        cn.add_vector_quantity('tangent', tangent, vectortype='ambient', radius=0.01)
        cn.add_vector_quantity('acceleration', acceleration, vectortype='ambient', radius=0.01)
        cn.add_vector_quantity('binormal', normalize(binormal), vectortype='ambient', radius=0.01)
        cn.add_vector_quantity('normal', normalize(normal), vectortype='ambient', radius=0.01)

        if self.show_curvature_comb:
            curvature = np.linalg.norm(binormal, axis=-1) / (np.linalg.norm(tangent, axis=-1) ** 3)
            curvature_vectors = curvature[..., None] * normalize(normal)
            curvature_comb_points = points + curvature_vectors

            ps.register_curve_network('curvature comb',
                                      nodes=np.vstack([points, curvature_comb_points]),
                                      edges=np.vstack([np.column_stack([np.arange(count),
                                                                        np.arange(count) + count]),
                                                       np.column_stack([np.arange(count-1)+count,
                                                                        np.arange(count-1)+count+1])]))
        else:
            ps.remove_curve_network('curvature comb', False)
