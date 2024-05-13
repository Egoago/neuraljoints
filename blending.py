import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Cube, Plane
from neuraljoints.neural.blending import BlendingNetwork, BlendingTrainer
from neuraljoints.ui.ui import UIHandler

if __name__ == "__main__":
    UIHandler.init()

    implicits = [aggregate.Union(name='A', children=[Cube()]),
                 aggregate.Union(name='B', children=[Plane()])]
    model = BlendingNetwork(implicits)
    trainer = BlendingTrainer(model=model, implicits=implicits)
    UIHandler.add_entity(trainer)
    for implicit in implicits:
        UIHandler.add_entity(implicit)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()
