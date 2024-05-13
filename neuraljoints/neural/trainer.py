import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import hessian, gradient
from neuraljoints.neural.losses import CompositeLoss, Mse
from neuraljoints.neural.model import Network
from neuraljoints.neural.sampling import PullSampler
from neuraljoints.neural.scheduler import LRScheduler
from neuraljoints.utils.parameters import IntParameter, FloatParameter


class Trainer(Entity):
    def __init__(self, model: Network, implicit: Implicit, **kwargs):
        implicit.name = 'Target'
        super().__init__(**kwargs)
        self.max_steps = IntParameter('max_steps', 10000, 1, 10000)
        self.lr = FloatParameter('lr', 1e-4, 1e-7, 0.1)

        self.model = model.to(self.device)
        self.sampler = PullSampler()
        self.implicit = implicit
        self.loss_fn = CompositeLoss(name='losses')
        self.loss_fn.add(Mse())
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr.value)
        self.scheduler = LRScheduler(self.optimizer, self.lr)
        self.training = False
        self.training_step = 0
        self.losses = []

    def train(self, step=0):
        self.training = True
        self.model = self.model.to(self.device)
        self.training_step = step
        if step == 0:
            self.losses = []

    def stop(self):
        self.training = False

    def reset(self):
        self.model.build()
        self.train()
        self.training = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr.value)  # TODO remove redundancy
        self.scheduler = LRScheduler(self.optimizer, self.lr)
        self.sampler.reset()

    def step(self):
        self.optimizer.zero_grad()
        outputs = {}

        x, surface_indices = self.sampler()
        outputs['x'] = x
        outputs['surface_indices'] = surface_indices
        x.requires_grad = self.loss_fn.req_grad

        outputs['y_gt'] = self.implicit(x)

        if 'grad_gt' in self.loss_fn.attributes:
            outputs['grad_gt'] = gradient(outputs['y_gt'], x)
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()

        x.requires_grad = self.loss_fn.req_grad or self.sampler.req_grad

        outputs['y_pred'] = self.model(x)

        if 'grad_pred' in self.loss_fn.attributes or self.sampler.req_grad:
            outputs['grad_pred'] = gradient(outputs['y_pred'], x)

        if 'hess_pred' in self.loss_fn.attributes:
            outputs['hess_pred'] = hessian(outputs['grad_pred'], x)

        outputs['loss'] = self.loss_fn(**outputs)
        return outputs

    def update(self):
        if self.training:
            self.training_step += 1

            outputs = self.step()

            outputs['loss'].backward()
            self.optimizer.step()
            self.scheduler.update()
            self.sampler.update(**outputs)

            self.losses.append(outputs['loss'].item())
            if self.training_step > self.max_steps.value:
                self.stop()
