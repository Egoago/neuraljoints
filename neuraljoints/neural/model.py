import math

import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.neural.embedding import NoEmbedding
from neuraljoints.utils.parameters import IntParameter, FloatParameter, ChoiceParameter


class Network(Entity, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = NoEmbedding()  # FrequencyEmbedding(in_dims=in_dims, freqs=2, add_position=True) TODO
        self.n_neurons = IntParameter('n_neurons', 512, 2, 512)
        self.n_layers = IntParameter('n_layers', 2, 1, 16)
        self.dropout = FloatParameter('dropout', 0., 0., 1.)
        self.mlp = self.build()

    def build(self):
        layers = []
        dim = self.embedding.out_dims
        for i in range(self.n_layers.value - 1):
            #layers.append(torch.nn.Linear(dim, self.n_neurons.value))
            #layers.append(torch.nn.ReLU())
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
