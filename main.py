import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Sphere
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.ui import UIHandler

if __name__ == "__main__":
    UIHandler.init()

    implicit = aggregate.Union(children=[Sphere()])
    model = Network()
    trainer = Trainer(model=model, implicit=implicit)
    UIHandler.add_entity(trainer)
    UIHandler.add_entity(implicit)


    while UIHandler.open:
        trainer.update()
        ps.frame_tick()
