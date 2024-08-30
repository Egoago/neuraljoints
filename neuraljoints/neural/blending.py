import torch

from neuraljoints.geometry.aggregate import Union, IPatchHierarchical
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient, hessian
from neuraljoints.neural.embedding import ImplicitEmbedding
from neuraljoints.neural.losses import MeanCurvature
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.utils.parameters import BoolParameter, FloatParameter


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

############################  |
# Optimized I-Patch blending  |
############################  v


class OptimizedFloatParameter(FloatParameter):
    initial: torch.Tensor
    value_: torch.Tensor

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def pre_root(cls, values: dict) -> dict:
        if 'value_' not in values and 'initial' in values:
            values['initial'] = torch.tensor(values['initial'], dtype=torch.float32)
            values['value_'] = values['initial'].clone()
        return values

    @FloatParameter.value.setter
    def value(self, value):
        self.value_ = torch.tensor(value, dtype=torch.float32).clamp(self.min, self.max)

    def to(self, device):
        self.value_ = self.value_.to(device)
        return self


class OptimizedIPatch(IPatchHierarchical):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w0 = OptimizedFloatParameter(name='w0', initial=self.w0.initial,
                                          min=self.w0.min, max=self.w0.max)
        self.exp = OptimizedFloatParameter(name='exp', initial=self.exp.initial,
                                           min=self.exp.min, max=self.exp.max)
        self.build_tree()

    def to(self, device):
        self.w0.to(device)
        self.exp.to(device)
        return self

    def build(self):
        self.w0.reset()
        self.exp.reset()

    def parameters(self):
        return [self.w0.value, self.exp.value]


class IPatchTrainer(Trainer):
    def __init__(self, model: OptimizedIPatch, **kwargs):
        super().__init__(model=model, implicit=None, **kwargs)
        self.loss_fn.empty()
        self.loss_fn.add(MeanCurvature())

    def step(self):
        outputs = self.sampler()
        x = outputs['x']
        x.requires_grad = self.loss_fn.req_grad or self.sampler.req_grad

        outputs['y_pred'] = self.model(x)

        if 'grad_pred' in self.loss_fn.attributes or 'grad_pred' in self.sampler.attributes:
            outputs['grad_pred'] = gradient(outputs['y_pred'], x)

        if 'hess_pred' in self.loss_fn.attributes:
            outputs['hess_pred'] = hessian(outputs['grad_pred'], x)

        outputs['loss'] = self.loss_fn(**outputs)
        return outputs
