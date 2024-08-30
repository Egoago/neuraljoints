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
        if implicit is not None:
            implicit.name = 'Target'
        super().__init__(**kwargs)
        self.max_steps = IntParameter(name='max_steps', initial=10000, min=1, max=10000)
        self.lr = FloatParameter(name='lr', initial=1e-4, min=1e-7, max=0.1)

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
        outputs = self.sampler()
        x = outputs['x']
        x.requires_grad = self.loss_fn.req_grad

        outputs['y_gt'] = self.implicit(x)

        if 'grad_gt' in self.loss_fn.attributes:
            outputs['grad_gt'] = gradient(outputs['y_gt'], x)

        x.requires_grad = self.loss_fn.req_grad or self.sampler.req_grad

        outputs['y_pred'] = self.model(x)

        if 'grad_pred' in self.loss_fn.attributes or 'grad_pred' in self.sampler.attributes:
            outputs['grad_pred'] = gradient(outputs['y_pred'], x)

        if 'hess_pred' in self.loss_fn.attributes:
            outputs['hess_pred'] = hessian(outputs['grad_pred'], x)

        outputs['loss'] = self.loss_fn(**outputs)
        return outputs

    def update(self):
        if self.training:
            self.training_step += 1
            self.optimizer.zero_grad()

            outputs = self.step()

            if torch.isnan(outputs['loss']).sum() > 0:
                print('NaN loss detected')
                self.stop()

            outputs['loss'].backward()
            self.optimizer.step()
            self.scheduler.update()
            self.sampler.update(**outputs)

            self.losses.append(outputs['loss'].item())
            if self.training_step > self.max_steps.value:
                self.stop()
