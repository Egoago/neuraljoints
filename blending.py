import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Cube, Plane, TransformedImplicit
from neuraljoints.neural.blending import BlendingNetwork, BlendingTrainer
from neuraljoints.ui.ui import UIHandler


def get_boundaries(implicit_a, implicit_b):
    boundaries = []
    for implicit in [implicit_b, implicit_a]:
        boundary = TransformedImplicit(child=implicit)
        boundary.offset.value = -0.5
        boundary.scale.value = -1.0
        boundaries.append(boundary)
    boundaries[0].name = f'{implicit_a.name} boundary'
    boundaries[1].name = f'{implicit_b.name} boundary'
    return boundaries


def main():
    UIHandler.init()

    implicits = [aggregate.Union(name='A', children=[Cube()]),
                 aggregate.Union(name='B', children=[Plane()])]
    boundaries = get_boundaries(implicits[0], implicits[1])
    model = BlendingNetwork(implicits, boundaries)
    trainer = BlendingTrainer(model=model)
    UIHandler.add_entity(trainer)
    for implicit in implicits:
        UIHandler.add_entity(implicit)
    for boundary in boundaries:
        UIHandler.add_entity(boundary)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()

if __name__ == "__main__":
    main()
