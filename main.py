import numpy as np
import polyscope as ps

from neuraljoints.geometry.aggregate import Union, RoundUnion
from neuraljoints.geometry.implicit import Cube
from neuraljoints.geometry.parametric import CubicBezier
import neuraljoints.geometry.parametric_to_implicit as p2i
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.ui import UIHandler

if __name__ == "__main__":
    UIHandler.init()

    cb = CubicBezier()
    cb.control_points.points = np.array([[-1, 0, 0],
                                         [-1, 1, 0],
                                         [1, 1, 0],
                                         [1, 0, 0]], dtype=np.float32)
    #implicit = p2i.ParametricToImplicitNewton(cb)
    implicit = RoundUnion(children=[Cube(), Cube()])
    model = Network()
    trainer = Trainer(model=model, implicit=implicit)
    UIHandler.add_entity(trainer)

    ps.set_user_callback(lambda: UIHandler.update())

    while True:
        trainer.update()
        ps.frame_tick()
