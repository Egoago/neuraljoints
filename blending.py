import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Cube, Plane, Offset
from neuraljoints.neural.blending import BlendingNetwork, BlendingTrainer
from neuraljoints.ui.ui import UIHandler


def get_boundaries(implicits):
    boundaries = []
    for implicit in implicits:
        # upper
        boundary = Offset(child=implicit)
        boundary.offset.value = 0.1
        boundaries.append(boundary)
        # lower
        boundary = Offset(child=implicit)
        boundary.offset.value = -0.1
        boundaries.append(boundary)
    return boundaries


def main():
    UIHandler.init()

    implicits = [aggregate.Union(name='A', children=[Cube()]),
                 aggregate.Union(name='B', children=[Plane()])]
    boundaries = get_boundaries(implicits)
    model = BlendingNetwork(implicits, boundaries)
    trainer = BlendingTrainer(model=model)
    UIHandler.add_entity(trainer)
    for implicit in implicits:
        UIHandler.add_entity(implicit)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()

if __name__ == "__main__":
    main()
