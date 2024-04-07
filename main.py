import numpy as np
import polyscope as ps

from neuraljoints.geometry.aggregate import Union, RoundUnion
from neuraljoints.geometry.implicit import Cube, Plane
from neuraljoints.geometry.parametric import CubicBezier
import neuraljoints.geometry.parametric_to_implicit as p2i
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.ui import UIHandler

if __name__ == "__main__":
    UIHandler.init()

    # cb1 = CubicBezier()
    # cb1.control_points.points = np.array([[-1, 0, 0],
    #                                       [-1, 1, 0],
    #                                       [1, 1, 0],
    #                                       [1, 0, 0]], dtype=np.float32)

    #implicit = p2i.ParametricToImplicitGWN(cb)
    implicit = RoundUnion(children=[Plane(),
                                    Plane()])
    #model = Network()
    #trainer = Trainer(model=model, implicit=implicit)
    #UIHandler.add_entity(trainer)
    UIHandler.add_entity(implicit)

    ps.set_user_callback(lambda: UIHandler.update())

    while True:
        #trainer.update()
        ps.frame_tick()
