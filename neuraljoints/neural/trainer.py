import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import hessian, gradient
from neuraljoints.neural.losses import CompositeLoss, Mse
from neuraljoints.neural.model import Network
from neuraljoints.neural.sampling import Sampler, ComplexSampler
from neuraljoints.neural.scheduler import LRScheduler
from neuraljoints.utils.parameters import IntParameter, FloatParameter


class Trainer(Entity):
    def __init__(self, model: Network, implicit: Implicit, **kwargs):
        implicit.name = 'Target'
        super().__init__(**kwargs)
        self.max_steps = IntParameter('max_steps', 10000, 1, 10000)
        self.lr = FloatParameter('lr', 5e-3, 1e-7, 0.1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.sampler = ComplexSampler()
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

    def update(self):
        if self.training:
            self.step += 1

            self.optimizer.zero_grad()
            y_grad, gradients, hessians = None, None, None
            with torch.no_grad():
                x = self.sampler()
                if self.loss_fn.req_grad:
                    y, y_grad = self.implicit(x, grad=True)
                    y_grad = torch.tensor(y_grad, device=self.device, dtype=torch.float32)
                else:
                    y = self.implicit(x)
                y = torch.tensor(y, device=self.device, dtype=torch.float32)
            x = torch.tensor(x, device=self.device, dtype=torch.float32, requires_grad=self.loss_fn.req_grad)

            pred = self.model(x)
            self.sampler.prev_y = pred.detach().cpu().numpy()

            if self.loss_fn.req_grad:
                gradients = gradient(pred, x)

            if self.loss_fn.req_hess:
                hessians = hessian(gradients, x)

            loss = self.loss_fn(x, y, pred, y_grad=y_grad, grad=gradients, hess=hessians)

            loss.backward()
            self.optimizer.step()
            self.scheduler.update()

            self.losses.append(loss.item())
            if self.step > self.max_steps.value:
                self.stop()
