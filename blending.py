import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Cube, Plane
from neuraljoints.neural.blending import OptimizedIPatch, IPatchTrainer
from neuraljoints.ui.ui import UIHandler


def main():
    UIHandler.init()

    implicits = [aggregate.Union(name='A', children=[Cube()]),
                 aggregate.Union(name='B', children=[Plane()])]
    model = OptimizedIPatch(children=implicits)
    trainer = IPatchTrainer(model=model)

    UIHandler.add_entity(trainer)
    UIHandler.add_entity(model)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()


if __name__ == "__main__":
    main()
