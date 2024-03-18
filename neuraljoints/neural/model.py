import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.neural.embedding import FrequencyEmbedding, NoEmbedding, Embedding2D
from neuraljoints.utils.parameters import IntParameter


class Network(Entity, torch.nn.Module):
    def __init__(self, in_dims=3, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding2D()#FrequencyEmbedding(in_dims=in_dims, freqs=2, add_position=True)
        self.n_neurons = IntParameter('neurons', 512, 2, 512)
        self.n_layers = IntParameter('layers', 3, 1, 16)
        self.mlp = self.build()

    def build(self):
        layers = [self.embedding, ]
        dim = self.embedding.out_dims
        for _ in range(self.n_layers.value-1):
            layers.append(torch.nn.Linear(dim, self.n_neurons.value))
            layers.append(torch.nn.ReLU())
            dim = self.n_neurons.value

        layers.append(torch.nn.Linear(dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        print(self.mlp)
        return self.mlp

    def forward(self, x: torch.Tensor):
        return self.mlp(x).squeeze()
