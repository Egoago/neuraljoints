from abc import abstractmethod

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.utils.parameters import IntParameter, Float3Parameter, FloatParameter


def detach_parameters(func):
    def wrapper(*args, **kwargs):
        detached_args = [arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in args]
        detached_kwargs = {key: val.detach() if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}
        return func(*detached_args, **detached_kwargs)
    return wrapper


class Pipeline(Entity):
    @property
    def attributes(self) -> set[str]:
        return set()

    def check_input(self, **kwargs):
        for attr in self.attributes:
            if attr not in kwargs:
                raise AttributeError(f'Input {attr} not found for {self.name}')

    @property
    def req_grad(self):
        return bool({'grad_gt', 'grad_pred', 'hess_pred'} & self.attributes)


class Sampler(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = IntParameter(name='batch_size', initial=8192 * 2, min=256, max=8192 * 2)
        self.bounds = Float3Parameter(name='bounds', initial=[2.1, 2.1, 2.1], min=0, max=4)
        self.surface_indices = torch.tensor([], dtype=torch.int, device=self.device)

    @torch.no_grad()
    def __call__(self):
        x = self._sample()
        bounds = self.bounds.value.to(device=self.device)
        x = torch.clamp(x, -bounds, bounds)

        return {'x': x,
                'surface_indices': self.surface_indices}

    @abstractmethod
    def _sample(self):
        return self.uniform_sample()

    def reset(self):
        self.surface_indices = torch.tensor([], dtype=torch.int, device=self.device)

    @abstractmethod
    @detach_parameters
    def update(self, **kwargs):
        self.check_input(**kwargs)

    def uniform_sample(self, count=None):
        if count is None:
            count = self.batch_size.value
        x = torch.rand((count, 3), dtype=torch.float32, device=self.device)
        x = (x * 2 - 1) * self.bounds.value[None, ...].to(device=self.device)
        return x


class HierarchicalSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_ratio = FloatParameter(name='surface ratio', initial=0.7, min=0., max=1.)
        self.noise_var = FloatParameter(name='noise var', initial=0.05, min=0., max=0.2)
        self.prev_y, self.prev_x = None, None

    def _sample(self):
        if (self.prev_y is None or self.prev_x is None or
                self.batch_size.value != len(self.prev_x)):
            self.surface_indices = torch.tensor([], dtype=torch.int, device=self.device)
            return Sampler._sample(self)

        indices = torch.argsort(torch.abs(self.prev_y))
        count = len(indices)
        surface_count = int(self.surface_ratio.value * count)
        self.surface_indices = indices[:surface_count]
        volume_indices = indices[surface_count:]

        x = self.prev_x

        x[volume_indices] = self.uniform_sample(len(volume_indices))
        noise = torch.randn((surface_count, 3), dtype=torch.float32, device=self.device)
        x[self.surface_indices] = x[self.surface_indices] + self.noise_var.value * noise

        return x

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'y_pred', 'x'}

    @detach_parameters
    def update(self, **kwargs):
        super().update(**kwargs)
        self.prev_y = kwargs['y_pred']
        self.prev_x = kwargs['x']

    def reset(self):
        super().reset()
        self.prev_y = None
        self.prev_x = None


class PullSampler(HierarchicalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_grad = None

    def _sample(self):
        x = super()._sample()
        if (self.prev_grad is not None and
            self.prev_y is not None and
            self.prev_x is not None):
            si = self.surface_indices
            x[si] = x[si] - self.prev_y[si][..., None] * self.prev_grad[si]
        return x

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'grad_pred'}

    @detach_parameters
    def update(self, **kwargs):
        super().update(**kwargs)
        self.prev_grad = kwargs['grad_pred']

    def reset(self):
        super().reset()
        self.prev_grad = None
