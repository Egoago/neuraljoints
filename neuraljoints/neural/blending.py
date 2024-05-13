from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.embedding import ImplicitEmbedding
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.utils.parameters import BoolParameter


class BlendingNetwork(Network):
    def __init__(self, implicits: list[Implicit], **kwargs):
        self.gradient = BoolParameter('use gradient', False)
        self.implicits = implicits
        super().__init__(**kwargs)

    def build(self):
        self.embedding = ImplicitEmbedding(self.implicits, self.gradient.value)
        return super().build()


class BlendingTrainer(Trainer):
    def __init__(self, model: Network, implicit: Implicit, **kwargs):
        super().__init__(model, implicit, **kwargs)