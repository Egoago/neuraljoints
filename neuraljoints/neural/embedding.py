from abc import abstractmethod

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient


class Embedding(torch.nn.Module, Entity):
    def __init__(self, in_dims=3):
        super().__init__()
        self.in_dims = in_dims

    @property
    @abstractmethod
    def out_dims(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class NoEmbedding(Embedding):
    @property
    def out_dims(self):
        return self.in_dims

    def forward(self, x):
        return x


class Embedding2D(Embedding):
    @property
    def out_dims(self):
        return self.in_dims-1

    def forward(self, x):
        return x[..., :-1]


class FrequencyEmbedding(Embedding):
    def __init__(self, freqs, add_position, **kwargs):
        super().__init__(**kwargs)
        self.freqs = freqs
        self.add_position = add_position

        self.register_buffer('k', torch.arange(freqs))
        self.register_buffer('freq_bands', 2**self.k * torch.pi)

    @property
    def out_dims(self):
        if self.add_position:
            return self.in_dims*(2*self.freqs+1)
        else:
            return self.in_dims*2*self.freqs

    def _spectrum(self, x):
        input_shape = x.shape                                               #[..., c]
        spectrum = x[..., None] * self.freq_bands                           #[..., c, L]
        spectrum = spectrum.reshape(*input_shape[:-1], -1)                  #[..., cL]
        spectrum = torch.cat([spectrum, spectrum+0.5*torch.pi], dim=-1)     #[..., 2cL]
        return spectrum

    def forward(self, x):
        spectrum = self._spectrum(x)
        if self.add_position:
            return torch.cat([x, torch.sin(spectrum)], -1) #[..., 2cL+c]
        else:
            return torch.sin(spectrum)                     #[..., 2cL]


class ImplicitEmbedding(Embedding):
    def __init__(self, implicits: list[Implicit], grad=False):
        super().__init__()
        self.implicits = implicits
        self.grad = grad

    @property
    def out_dims(self):
        return len(self.implicits) * (4 if self.grad else 1)

    def forward(self, x):
        if self.grad and not x.requires_grad:
            x.requires_grad = True

        values, gradients = [], []
        for implicit in self.implicits:
            value = implicit(x)
            values.append(value)
            if self.grad:
                gradients.append(gradient(value, x))

        x = torch.stack(values, dim=-1)
        if self.grad:
            gradients = torch.concatenate(gradients, dim=-1)
            x = torch.concatenate([x, gradients], dim=-1)
        return x
