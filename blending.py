import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Cube, Plane
from neuraljoints.neural.model import BlendingNetwork
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.ui import UIHandler

if __name__ == "__main__":
    UIHandler.init()

    implicits = [aggregate.Union(children=[Cube()]),
                 aggregate.Union(children=[Plane()])]
    model = BlendingNetwork(implicits)
    trainer = Trainer(model=model, implicits=implicits)
    UIHandler.add_entity(trainer)
    for implicit in implicits:
        UIHandler.add_entity(implicit)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()
