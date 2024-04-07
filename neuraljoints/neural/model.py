import math

import torch
import torch.functional as F
import tinycudann as tcnn

from neuraljoints.geometry.base import Entity
from neuraljoints.neural.embedding import FrequencyEmbedding, NoEmbedding, Embedding2D
from neuraljoints.utils.parameters import IntParameter, FloatParameter, ChoiceParameter


class Network(Entity, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding2D()  # FrequencyEmbedding(in_dims=in_dims, freqs=2, add_position=True)
        self.n_neurons = IntParameter('n_neurons', 512, 2, 512)
        self.n_layers = IntParameter('n_layers', 2, 1, 16)
        self.dropout = FloatParameter('dropout', 0., 0., 1.)
        self.mlp = self.build()

    def build(self):
        layers = []
        dim = self.embedding.out_dims
        for i in range(self.n_layers.value - 1):
            # layers.append(torch.nn.Linear(dim, self.n_neurons.value))
            # layers.append(torch.nn.ReLU())
            layers.append(Siren(dim, self.n_neurons.value, i == 0))
            dim = self.n_neurons.value
        if self.dropout.value > 0:
            layers.append(torch.nn.Dropout(self.dropout.value))
        layers.append(torch.nn.Linear(dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        print(self.mlp)
        return self.mlp

    def forward(self, x):
        x = self.embedding(x)
        return self.mlp(x).squeeze()


class Sine(torch.nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(torch.nn.Module):
    def __init__(self, dim_in, dim_out, is_first=False, c=0.5, use_bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        w0 = 30. if is_first else 1.

        self.linear = torch.nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = Sine(w0)

        self.init_params(c, w0)

    def init_params(self, c, w0):
        dim = self.dim_in
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        torch.nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if self.linear.bias is not None:
            torch.nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class HashMLP(Network):
    def __init__(self, **kwargs):
        self.n_levels = IntParameter('n_levels', 4, 1, 16)
        self.n_features_per_level = IntParameter('n_features_per_level', 2, 1, 2)
        self.log2_hashmap_size = IntParameter('log2_hashmap_size', 10, 2, 15)
        self.base_resolution = IntParameter('base_resolution', 8, 2, 16)
        self.per_level_scale = FloatParameter('per_level_scale', 1.5, 1.5, 2)
        self.activation = ChoiceParameter('activation', "ReLU", ["None", "ReLU", "LeakyReLU",
                                                                 "Exponential", "Sine", "Sigmoid",
                                                                 "Squareplus", "Softplus", "Tanh"])
        super().__init__(**kwargs)

    def build(self):
        encoding = {"otype": "HashGrid",
                    "n_levels": self.n_levels.value,
                    "n_features_per_level": self.n_features_per_level.value,
                    "log2_hashmap_size": self.log2_hashmap_size.value,
                    "base_resolution": self.base_resolution.value,
                    "per_level_scale": self.per_level_scale.value}
        network = {"otype": "FullyFusedMLP",
                   "activation": self.activation.value,
                   "output_activation": "None",
                   "n_neurons": self.n_neurons.value,
                   "n_hidden_layers": self.n_layers.value}
        self.mlp = tcnn.NetworkWithInputEncoding(self.embedding.out_dims, 1,
                                                 encoding, network)
        print(self.mlp)
        return self.mlp
