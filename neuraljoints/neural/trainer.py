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
        self.step = 0
        self.losses = []

    def train(self, step=0):
        self.training = True
        self.model = self.model.to(self.device)
        self.step = step
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

    def update(self):
        if self.training:
            self.step += 1

            self.optimizer.zero_grad()
            y_grad, gradients, hessians = None, None, None

            x, surface_indices = self.sampler()
            x.requires_grad = self.loss_fn.req_grad

            y = self.implicit(x)

            if self.loss_fn.req_grad:
                y_grad = gradient(y, x)
                if x.grad is not None:
                    x.grad.detach_()
                    x.grad.zero_()

            x.requires_grad = self.loss_fn.req_grad or self.sampler.req_grad

            pred = self.model(x)

            if self.loss_fn.req_grad or self.sampler.req_grad:
                gradients = gradient(pred, x)

            if self.loss_fn.req_hess:
                hessians = hessian(gradients, x)

            loss = self.loss_fn(x, y, pred,
                                y_grad=y_grad, grad=gradients, hess=hessians,
                                surface_indices=surface_indices)

            loss.backward()
            self.optimizer.step()
            self.scheduler.update()
            self.sampler.update(prev_y=pred, prev_x=x, prev_grad=gradients)

            self.losses.append(loss.item())
            if self.step > self.max_steps.value:
                self.stop()
