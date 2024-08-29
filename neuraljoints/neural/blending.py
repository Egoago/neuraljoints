from neuraljoints.geometry.aggregate import Union
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.embedding import ImplicitEmbedding
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.utils.parameters import BoolParameter


class BlendingNetwork(Network):
    def __init__(self, implicits: list[Implicit], boundaries: list[Implicit], **kwargs):
        self.gradient = BoolParameter(name='use gradient', initial=False)
        self.use_boundaries = BoolParameter(name='use boundaries', initial=False)
        self.implicits = implicits
        self.boundaries = boundaries
        super().__init__(**kwargs)

    def build(self):
        inputs = self.implicits
        if self.use_boundaries.value:
            inputs = inputs + self.boundaries
        self.embedding = ImplicitEmbedding(inputs, self.gradient.value)
        return super().build()


class BlendingTrainer(Trainer):
    def __init__(self, model: BlendingNetwork, **kwargs):
        super().__init__(model=model, implicit=Union(name='Union', children=model.implicits), **kwargs)
