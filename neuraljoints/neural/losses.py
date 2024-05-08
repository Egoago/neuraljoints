from abc import abstractmethod, ABC

import torch

from neuraljoints.geometry.base import Entity, Set
from neuraljoints.utils.parameters import FloatParameter


class Loss(Entity):
    _req_grad = False
    _req_hess = False
    _on_surface = False
    _on_volume = False

    @property
    def req_grad(self):
        return (self._req_grad or self._req_hess) and self.enabled

    @property
    def req_hess(self):
        return self._req_hess and self.enabled

    @property
    def enabled(self):
        return True

    def __call__(self, x, y, pred, y_grad=None, grad=None, hess=None, surface_indices=None):
        if self.req_hess:
            assert hess is not None
        if self.req_grad:
            assert grad is not None
        if self._on_surface or self._on_volume:
            assert surface_indices is not None
            index = surface_indices
            if self._on_volume:
                index = torch.ones(len(x), dtype=torch.bool, device=x.device)
                index[surface_indices] = False
            x, y, pred, y_grad, grad, hess = [arg[index]
                                              for arg in [x, y, pred, y_grad, grad, hess]
                                              if arg is not None]

        if self.enabled:
            return self.energy(x=x, y=y, pred=pred,
                               y_grad=y_grad, grad=grad, hess=hess,
                               surface_indices=surface_indices).mean()
        return 0.

    @abstractmethod
    def energy(self, *args, **kwargs):
        raise NotImplementedError()


class WeightedLoss(Loss, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = FloatParameter('', 1, 0, 1)

    def __call__(self, *args, **kwargs):
        return Loss.__call__(self, *args, **kwargs) * self.weight.value


class CompositeLoss(Loss, Set):
    def req_grad2(self):
        for loss in self.children:
            if loss.req_grad:
                return True
        return False

    @property
    def losses(self) -> set[Loss]:
        return self.children

    @property
    def req_grad(self):
        for loss in self.children:
            if loss.req_grad:
                return True
        return False

    @property
    def req_hess(self):
        for loss in self.children:
            if loss.req_hess:
                return True
        return False

    def energy(self, *args, **kwargs):
        sum = 0.
        for loss in self.children:
            sum = sum + loss(*args, **kwargs)
        return sum


class Mse(WeightedLoss):
    def energy(self, y, pred, *args, **kwargs):
        return (y - pred) ** 2


class Eikonal(WeightedLoss):
    _req_grad = True

    def energy(self, grad, *args, **kwargs):
        grad_norm = torch.linalg.norm(grad, axis=-1)
        return (grad_norm - 1).abs()


class EikonalRelaxed(WeightedLoss):
    _req_grad = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_threshold = FloatParameter('grad threshold', 0.8, 0, 1.)
        self.relu = torch.nn.ReLU()

    def energy(self, grad, *args, **kwargs):
        grad_norm = torch.linalg.norm(grad, axis=-1)
        return self.relu(self.grad_threshold.value - grad_norm)


class Dirichlet(WeightedLoss):
    _req_grad = True
    _on_surface = True

    def energy(self, grad, *args, **kwargs):
        return (grad ** 2).sum(dim=-1)


class Neumann(WeightedLoss):
    _req_grad = True

    def energy(self, y_grad, grad, *args, **kwargs):
        grad = torch.nn.functional.normalize(grad, dim=-1)
        return 1 - (y_grad * grad).sum(dim=-1)


class Laplacian(WeightedLoss):
    _req_hess = True
    _on_surface = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def energy(self, hess, *args, **kwargs):
        trace = torch.einsum("...ii", hess)  # trace of the hessian is the laplacian

        return trace


class HessianDet(WeightedLoss):
    _req_hess = True

    def energy(self, hess, *args, **kwargs):
        determinants = torch.linalg.det(hess)
        return torch.abs(determinants)


class GaussianCurvature(WeightedLoss):
    _req_hess = True
    _on_surface = True

    def curvature(self, hess, grad):
        mat = torch.cat([hess, grad[..., None]], -1)
        row = torch.cat([grad, torch.zeros_like(grad[..., 0])[..., None]], -1)
        mat = torch.cat([mat, row[..., None, :]], -2)
        determinants = torch.linalg.det(mat)
        norm_4 = (grad ** 2).sum(dim=-1) ** 2
        return - determinants / norm_4

    def energy(self, grad, hess, *args, **kwargs):
        curvature = self.curvature(hess, grad)
        return curvature.abs()


class DoubleThrough(GaussianCurvature):

    @staticmethod
    def double_through(x):
        pi = torch.pi
        return (64 * pi - 80) / (pi ** 4) * (x ** 4) - (64 * pi - 88) / (pi ** 3) * (x ** 3) + (16 * pi - 29) / (
                    pi ** 2) * (x ** 2) + 3 * x / pi

    def energy(self, *args, **kwargs):
        energy = super().energy(*args, **kwargs)
        return DoubleThrough.double_through(energy)


class HessianAlign(WeightedLoss):
    _req_grad = True
    _req_hess = True
    _on_surface = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = FloatParameter('delta', 10., 0.1, 100)

    def energy(self, pred, grad, hess, *args, **kwargs):
        grad = torch.nn.functional.normalize(grad, dim=-1)
        energy = torch.matmul(hess, grad.unsqueeze(-1)).squeeze().square().sum(dim=-1)

        weights = torch.exp(-self.delta.value * torch.abs(pred))
        return energy * weights
