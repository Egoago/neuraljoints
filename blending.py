import polyscope as ps

import neuraljoints.geometry.aggregate as aggregate
from neuraljoints.geometry.implicit import Plane
from neuraljoints.neural import blending
from neuraljoints.ui.ui import UIHandler


def main():
    UIHandler.init()
    p = Plane()
    p.transform.rotation.value = [0, 0, -90]
    implicits = [aggregate.Union(name='A', children=[p]),
                 aggregate.Union(name='B', children=[Plane()])]
    model = blending.OptimizedIPatchProductum(children=implicits)
    trainer = blending.IPatchTrainer(model=model)

    UIHandler.add_entity(trainer)
    UIHandler.add_entity(model)

    while UIHandler.open:
        trainer.update()
        ps.frame_tick()


if __name__ == "__main__":
    main()
