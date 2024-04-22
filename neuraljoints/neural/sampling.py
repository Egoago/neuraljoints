import numpy as np

from neuraljoints.geometry.base import Entity
from neuraljoints.utils.parameters import IntParameter, Float3Parameter, FloatParameter


class Sampler(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = IntParameter('batch_size', 8192*2, 256, 8192*2)
        self.bounds = Float3Parameter('bounds', value=[2.1, 2.1, 2.1], min=0, max=4)
        self.prev_x = None
        self.prev_y = None

    def __call__(self):
        self.prev_x = self.sample()
        return self.prev_x

    def sample(self, count=None):
        if count is None:
            count = self.batch_size.value
        x = np.random.rand(count, 3).astype(dtype=np.float32)
        x = (x * 2 - 1) * self.bounds.value[None, ...]
        return x.astype(dtype=np.float32)


class ComplexSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_ratio = FloatParameter('surface ratio', 0.5, 0., 1.)
        self.noise_var = FloatParameter('noise var', 0.1, 1e-5, 1.)

    def __call__(self):
        if self.prev_y is None or self.batch_size.value != len(self.prev_x):
            return Sampler.__call__(self)

        indices = np.argsort(np.abs(self.prev_y))
        count = len(indices)
        surface_count = int(self.surface_ratio.value * count)
        volume_count = count - surface_count
        self.prev_x[indices[surface_count:]] = self.sample(volume_count)
        self.prev_x[indices[:surface_count]] = self.prev_x[indices[:surface_count]] + np.random.randn(surface_count, 3) * self.noise_var.value
        self.prev_x = np.clip(self.prev_x, -self.bounds.value, self.bounds.value)
        return self.prev_x
