import math
from abc import abstractmethod

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.neural.embedding import NoEmbedding
from neuraljoints.utils.parameters import IntParameter, ChoiceParameter, Parameter, FloatParameter
from neuraljoints.utils.utils import RegisteredMeta


class Network(Entity, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = NoEmbedding()  # TODO
        self.n_neurons = IntParameter('n_neurons', 128, 2, 512)
        self.n_layers = IntParameter('n_layers', 1, 0, 8)
        self.layer = ChoiceParameter('architecture', Siren.__name__, list(Layer.subclass_names))
        self.init_scheme = ChoiceParameter('init', Siren.init_schemes[0], ReLU.init_schemes)
        self.mlp = self.build()
        self.layers = []

    @property
    def hparams(self):
        attributes = list(self.__dict__.values()) + list(Layer.get_subclass(self.layer.value).__dict__.values())
        return [v for v in attributes if isinstance(v, Parameter)]

    def build(self):
        self.layers = []
        in_dim = self.embedding.out_dims
        out_dim = self.n_neurons.value
        LayerType = Layer.get_subclass(self.layer.value)
        for i in range(self.n_layers.value):
            layer = LayerType(in_dim=in_dim, out_dim=out_dim,
                              index=i, layer_count=self.n_layers.value)
            layer.init(self.init_scheme.value)
            self.layers.append(layer)
            in_dim = out_dim

        linear = torch.nn.Linear(in_dim, 1)
        linear = LayerType.init_linear(linear, self.init_scheme.value)
        self.layers.append(linear)

        self.mlp = torch.nn.Sequential(*self.layers)
        print(self.mlp)
        return self.mlp

    def forward(self, x):
        x = self.embedding(x)
        return self.mlp(x).squeeze()


class Layer(torch.nn.Module, metaclass=RegisteredMeta):
    def __init__(self, in_dim, out_dim, index, layer_count, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.index = index
        self.layer_count = layer_count

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.layer_count - 1

    @classmethod
    @property
    @abstractmethod
    def init_schemes(cls) -> list[str]:
        return ['']

    @abstractmethod
    def init(self, scheme: str):
        raise NotImplementedError(f'Initialization scheme {scheme} is not ' +
                                  f'supported for layer {self.__class__.__name__}')

    @classmethod
    def init_linear(cls, linear: torch.nn.Linear, scheme: str) -> torch.nn.Module:
        return linear

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class LinearLayer(Layer):
    def __init__(self, in_dim, out_dim, **kwargs):
        if 'activation' not in kwargs:
            kwargs['activation'] = torch.nn.Identity()
        super().__init__(in_dim, out_dim, **kwargs)

    @torch.no_grad()
    def init_uniform(self, std=None, bias_std=None):
        if std is None:
            std = 1. / math.sqrt(self.in_dim)
        torch.nn.init.uniform_(self.linear.weight, -std, std)
        if self.linear.bias is not None:
            if bias_std is None:
                bias_std = std
            torch.nn.init.uniform_(self.linear.bias, -bias_std, bias_std)

    @classmethod
    @property
    def init_schemes(cls) -> list[str]:
        return ['uniform']

    def init(self, scheme: str):
        if scheme == 'uniform':
            self.init_uniform()
        else:
            super().init(scheme)


class ReLU(LinearLayer):
    def __init__(self, in_dim, out_dim, **kwargs):
        kwargs['activation'] = torch.nn.ReLU()
        super().__init__(in_dim, out_dim, **kwargs)


class SoftPlus(LinearLayer):
    BETA = FloatParameter('beta', 5., 1e-5, 10.)

    def __init__(self, in_dim, out_dim, **kwargs):
        kwargs['activation'] = torch.nn.Softplus(self.BETA.value)
        super().__init__(in_dim, out_dim, **kwargs)


class Sine(torch.nn.Module):
    W0 = FloatParameter('frequency', 30., 1., 30.)

    def forward(self, x):
        return torch.sin(Sine.W0.value * x)


class Siren(LinearLayer):
    MFGI_RADIUS = FloatParameter('mfgi radius', 1., 1e-3, 4.)
    C = 6.

    def __init__(self, in_dim, out_dim, **kwargs):
        kwargs['activation'] = Sine()
        super().__init__(in_dim, out_dim, **kwargs)

    def init_uniform(self, std=None, bias_std=None):
        std = math.sqrt(Siren.C / self.in_dim) / Sine.W0.value
        super().init_uniform(std)

    def init_uniform_first(self):
        super().init_uniform(1. / self.in_dim)

    def init_geom(self):
        std = math.sqrt(3. / self.out_dim) / Sine.W0.value
        bias_std = 1. / (self.out_dim * 1000 * Sine.W0.value)
        super().init_uniform(std, bias_std)

    @torch.no_grad()
    def init_geom_last(self):
        assert self.in_dim == self.out_dim
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype
        self.linear.weight.data = (0.5 * torch.pi * torch.eye(self.out_dim, device=device, dtype=dtype) +
                                   0.001 * torch.randn(self.out_dim, self.out_dim, device=device, dtype=dtype)) / Sine.W0.value
        if self.linear.bias is not None:
            self.linear.bias.data = (0.5 * torch.pi * torch.ones(self.out_dim, device=device, dtype=dtype) +
                                     0.001 * torch.randn(self.out_dim, device=device, dtype=dtype)) / Sine.W0.value

    @torch.no_grad()
    def init_mfgi_first(self):
        self.init_geom()
        index = int(self.out_dim * 0.25)
        self.linear.weight.data[index:] = self.linear.weight.data[index:] * Sine.W0.value

    @torch.no_grad()
    def init_mfgi_second(self):
        self.init_geom()
        index = int(self.out_dim * 0.25)
        original = self.linear.weight.data[:index, :index]
        self.linear.weight.data = self.linear.weight.data * 1e-3
        self.linear.weight.data[:index, :index] = original

    @classmethod
    @torch.no_grad()
    def init_geom_linear(cls, linear: torch.nn.Linear):
        num_input = linear.weight.size(-1)
        assert linear.weight.shape == (1, num_input)
        assert linear.bias.shape == (1,)
        linear.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
        linear.bias.data = num_input + 0.0001 * torch.randn(1)

    @classmethod
    @property
    def init_schemes(cls) -> list[str]:
        return ['uniform siren', 'geometric', 'mfgi'] + super().init_schemes

    def init(self, scheme: str):
        if scheme == 'uniform siren':
            if self.is_first:
                self.init_uniform_first()
            else:
                self.init_uniform()
        elif scheme == 'geometric':
            if self.is_first:
                self.init_geom()
            elif self.is_last:
                self.init_geom_last()
            else:
                self.init_geom()
        elif scheme == 'mfgi':
            if self.is_first:
                self.init_mfgi_first()
            elif self.index == 1:
                self.init_mfgi_second()
            elif self.is_last:
                self.init_geom_last()
            else:
                self.init_geom()

        else:
            super().init(scheme)

    @classmethod
    def init_linear(cls, linear: torch.nn.Linear, scheme: str) -> torch.nn.Module:
        if scheme in ['geometric', 'mfgi']:
            cls.init_geom_linear(linear)

            class GeomLast(torch.nn.Module):
                def __init__(self, linear, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.linear = linear

                def forward(self, x):
                    x = self.linear(x)
                    x = torch.sign(x) * torch.sqrt(x.abs() + 1e-8)
                    x = x - Siren.MFGI_RADIUS.value
                    return x
            return GeomLast(linear)
        return super().init_linear(linear, scheme)
