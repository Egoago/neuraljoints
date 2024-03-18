import torch

from neuraljoints.geometry.base import Entity
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.model import Network
from neuraljoints.neural.sampling import Sampler
from neuraljoints.neural.scheduler import LRScheduler
from neuraljoints.utils.parameters import IntParameter, FloatParameter


class Trainer(Entity):
    def __init__(self, model: Network, implicit: Implicit, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = IntParameter('max_steps', 1000, 1, 10000)
        self.lr = FloatParameter('lr', 1e-2, 1e-7, 0.1)
        self.eikonal_loss = FloatParameter('eikonal_loss', 1e-2, 0, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.sampler = Sampler()
        self.implicit = implicit
        self.loss_fn = torch.nn.MSELoss()
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

    @property
    def calculate_eikonal(self):
        return self.eikonal_loss.value > 0

    def update(self):
        if self.training:
            self.step += 1

            self.optimizer.zero_grad()
            with torch.no_grad():
                x = self.sampler()
                y = torch.tensor(self.implicit(x), device=self.device, dtype=torch.float32)
            x = torch.tensor(x, device=self.device, dtype=torch.float32, requires_grad=self.calculate_eikonal)

            pred = self.model(x)
            if self.calculate_eikonal:
                pred.sum().backward(inputs=x, retain_graph=True)
                grads = x.grad.detach()
                eikonal_loss = ((torch.linalg.norm(grads, axis=-1) - 1)**2).mean()
            loss = self.loss_fn(pred, y)
            if self.calculate_eikonal:
                loss = loss + self.eikonal_loss.value * eikonal_loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.update()

            self.losses.append(loss.item())
            if self.step > self.max_steps.value:
                self.stop()
