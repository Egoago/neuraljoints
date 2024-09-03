import torch

from neuraljoints.geometry.aggregate import Union, IPatchHierarchical, IPatchPair
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural import losses
from neuraljoints.neural.autograd import gradient, hessian
from neuraljoints.neural.embedding import ImplicitEmbedding
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


class OptimizedIPatchPair(IPatchPair):
    def blend(self, implicits, boundaries):
        boundaries = boundaries.abs() ** self.exp     # handle exp and w0 as tensors, not params
        b_prod = boundaries.prod(dim=0)
        dividend, divisor = -self.w0 * b_prod, 0
        for i in range(len(implicits)):
            temp = b_prod / boundaries[i]
            dividend += temp * implicits[i]
            divisor += temp
        return dividend / divisor


class OptimizedIPatch(IPatchHierarchical):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w0_param: FloatParameter = self.w0     # switch parameters to tensors
        self.exp_param: FloatParameter = self.exp
        self.w0 = torch.tensor(self.w0_param.value, requires_grad=True)
        self.exp = torch.tensor(self.exp_param.value, requires_grad=True)

        self.build_tree()

    def to(self, device):
        self.w0 = torch.tensor(self.w0_param.value, requires_grad=True, device=device)
        self.exp = torch.tensor(self.exp_param.value, requires_grad=True, device=device)
        self.build_tree()
        return self

    def update(self, training):
        if training:
            self.w0_param.value = self.w0.item()
            self.exp_param.value = self.exp.item()
            self.w0_param.changed = True
            self.exp_param.changed = True
        else:
            if self.w0.item() != self.w0_param.value:
                self.w0 = torch.tensor(self.w0_param.value)
                self.w0_param.changed = True
            if self.exp.item() != self.exp_param.value:
                self.exp = torch.tensor(self.exp_param.value)
                self.exp_param.changed = True
            if self.w0_param.changed or self.exp_param.changed:
                self.build_tree()

    def build(self):
        self.w0_param.reset()
        self.exp_param.reset()
        self.w0_param.changed = True
        self.exp_param.changed = True
        self.w0 = torch.tensor(self.w0_param.value)
        self.exp = torch.tensor(self.exp_param.value)
        self.build_tree()

    def build_tree(self):
        self.tree = None
        if len(self.children) > 0:
            self.tree = self.children[0]
            for implicit in self.children[1:]:
                self.tree = OptimizedIPatchPair(implicit_a=self.tree, implicit_b=implicit,
                                                offset=self.offset, exp=self.exp, w0=self.w0)

    def parameters(self):
        return [self.w0, self.exp]


class OptimizedIPatchPairProductum(IPatchPair):
    def __init__(self, exp_offset, w0_offset, **kwargs):
        super().__init__(**kwargs)
        self.exp_offset = exp_offset
        self.w0_offset = w0_offset

    def blend(self, implicits, boundaries):
        boundaries = boundaries.abs() ** 2
        b_prod = boundaries.prod(dim=0)
        exp = b_prod * self.exp + self.exp_offset
        w0 = b_prod * self.w0 + self.w0_offset

        boundaries = boundaries ** exp
        dividend, divisor = -w0 * b_prod, 0
        for i in range(len(implicits)):
            temp = b_prod / boundaries[i]
            dividend += temp * implicits[i]
            divisor += temp
        return dividend / divisor


class OptimizedIPatchProductum(IPatchHierarchical):
    def __init__(self, **kwargs):
        self.w0_offset_param = FloatParameter(name='w0 offset', initial=1., min=-1., max=10.)
        self.exp_offset_param = FloatParameter(name='exp offset', initial=1., min=0., max=10.)
        self.w0_offset = torch.tensor(self.w0_offset_param.value, requires_grad=True)
        self.exp_offset = torch.tensor(self.exp_offset_param.value, requires_grad=True)
        super().__init__(**kwargs)
        self.w0_param: FloatParameter = self.w0  # switch parameters to tensors
        self.exp_param: FloatParameter = self.exp
        self.w0 = torch.tensor(self.w0_param.value, requires_grad=True)
        self.exp = torch.tensor(self.exp_param.value, requires_grad=True)
        self.w0_param.reset(initial=1e-6)
        self.exp_param.reset(initial=1e-6)
        self.w0_param.min = -10.
        self.exp_param.min = -10.
        self.build_tree()

    def to(self, device):
        self.w0 = torch.tensor(self.w0_param.value, requires_grad=True, device=device)
        self.exp = torch.tensor(self.exp_param.value, requires_grad=True, device=device)
        self.w0_offset = torch.tensor(self.w0_offset_param.value, requires_grad=True, device=device)
        self.exp_offset = torch.tensor(self.exp_offset_param.value, requires_grad=True, device=device)

        self.build_tree()
        return self

    def update(self, training):
        if training:
            self.w0_param.value = self.w0.item()
            self.exp_param.value = self.exp.item()
            self.w0_offset_param.value = self.w0_offset.item()
            self.exp_offset_param.value = self.exp_offset.item()
            self.w0_param.changed = True
            self.exp_param.changed = True
            self.w0_offset_param.changed = True
            self.exp_offset_param.changed = True
        else:
            if self.w0.item() != self.w0_param.value:
                self.w0 = torch.tensor(self.w0_param.value)
                self.w0_param.changed = True
            if self.exp.item() != self.exp_param.value:
                self.exp = torch.tensor(self.exp_param.value)
                self.exp_param.changed = True
            if self.w0_offset.item() != self.w0_offset_param.value:
                self.w0 = torch.tensor(self.w0_offset_param.value)
                self.w0_offset_param.changed = True
            if self.exp_offset.item() != self.exp_offset_param.value:
                self.exp_offset = torch.tensor(self.exp_offset_param.value)
                self.exp_offset_param.changed = True
            if self.w0_param.changed or self.exp_param.changed or self.w0_offset_param.changed or self.exp_offset_param.changed:
                self.build_tree()

    def build(self):
        self.w0_param.reset()
        self.exp_param.reset()
        self.w0_offset_param.reset()
        self.exp_offset_param.reset()
        self.w0_param.changed = True
        self.exp_param.changed = True
        self.w0_offset_param.changed = True
        self.exp_offset_param.changed = True
        self.w0 = torch.tensor(self.w0_param.value)
        self.exp = torch.tensor(self.exp_param.value)
        self.w0_offset = torch.tensor(self.w0_offset_param.value)
        self.exp_offset = torch.tensor(self.exp_offset_param.value)
        self.build_tree()

    def build_tree(self):
        self.tree = None
        if len(self.children) > 0:
            self.tree = self.children[0]
            for implicit in self.children[1:]:
                self.tree = OptimizedIPatchPairProductum(implicit_a=self.tree, implicit_b=implicit,
                                                         offset=self.offset, exp=self.exp, w0=self.w0,
                                                         exp_offset=self.exp_offset, w0_offset=self.w0_offset)

    def parameters(self):
        return [self.w0, self.exp, self.w0_offset, self.exp_offset]


class IPatchTrainer(Trainer):
    def __init__(self, model: OptimizedIPatch, **kwargs):
        super().__init__(model=model, implicit=None, **kwargs)
        self.loss_fn.empty()
        self.loss_fn.add(losses.MeanCurvature())
        self.detect_anomaly = BoolParameter(name='detect anomaly', initial=False)

    def step(self):
        torch.autograd.set_detect_anomaly(self.detect_anomaly.value)

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

    def update(self):
        super().update()
        if self.training:
            for i, param in enumerate(self.model.parameters()):
                if torch.isnan(param).sum() > 0:
                    print(f'NaN parameter detected at {i}th index')
                    self.stop()
        self.model.update(self.training)
