import numpy as np
import polyscope as ps

from neuraljoints.geometry.parametric import Parametric
from neuraljoints.ui.wrappers.control_point_wrapper import ControlPointsWrapper
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper


class ParametricWrapper(EntityWrapper):

    RESOLUTION = 100

    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(entity=parametric, **kwargs)

    @property
    def parametric(self) -> Parametric:
        return self.entity

    def draw_geometry(self):
        super().draw_geometry()
        parameters = np.linspace(0., 1., self.RESOLUTION, dtype=np.float32)
        points = self.parametric(parameters)
        ps.register_curve_network(self.parametric.name, points, 'line')
