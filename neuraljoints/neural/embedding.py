from abc import abstractmethod

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.utils.parameters import IntParameter


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
        return


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
