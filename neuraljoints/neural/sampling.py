import numpy as np

from neuraljoints.geometry.base import Entity
from neuraljoints.utils.parameters import IntParameter, Float3Parameter


class Sampler(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = IntParameter('batch_size', 8192*2, 256, 8192*2)
        self.bounds = Float3Parameter('bounds', value=[4, 4, 0], min=0, max=4)

    def __call__(self):
        x = np.random.rand(self.batch_size.value, 3).astype(dtype=np.float32)
        x = (x*2-1) * self.bounds.value[None, ...]
        return x.astype(dtype=np.float32)
