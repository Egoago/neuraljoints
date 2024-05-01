from abc import abstractmethod, ABC

import torch

from neuraljoints.geometry.base import Entity, Set
from neuraljoints.utils.parameters import FloatParameter


class Loss(Entity):
    _req_grad = False
    _req_hess = False

    @property
    def req_grad(self):
        return (self._req_grad or self._req_hess) and self.enabled

    @property
    def req_hess(self):
        return self._req_hess and self.enabled

    @property
    def enabled(self):
        return True

    def __call__(self, x, y, pred, y_grad=None, grad=None, hess=None):
        if self.req_hess:
            assert hess is not None
        if self.req_grad:
            assert grad is not None

        if self.enabled:
            return self.energy(x, y, pred, y_grad, grad, hess).mean()
        return 0.

    @abstractmethod
    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
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
    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        return (y - pred) ** 2


class Eikonal(WeightedLoss):
    _req_grad = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        grad_norm = torch.linalg.norm(grad, axis=-1)
        return (grad_norm - 1).abs()


class EikonalRelaxed(WeightedLoss):
    _req_grad = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_threshold = FloatParameter('grad threshold', 0.8, 0, 1.)
        self.relu = torch.nn.ReLU()

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        grad_norm = torch.linalg.norm(grad, axis=-1)
        return self.relu(self.grad_threshold.value - grad_norm)


class Dirichlet(WeightedLoss):
    _req_grad = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        return (grad ** 2).sum(dim=-1)


class Neumann(WeightedLoss):
    _req_grad = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        grad = torch.nn.functional.normalize(grad, dim=-1)
        return (1 - (y_grad * grad).sum(dim=-1))


class Laplacian(WeightedLoss):
    _req_hess = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        trace = torch.einsum("...ii", hess) #trace of the hessian is the laplacian

        return trace


class Hessian(WeightedLoss):
    _req_hess = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        raise NotImplementedError()


class HessianDet(WeightedLoss):
    _req_hess = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        determinants = torch.linalg.det(hess)
        return torch.abs(determinants)


class GaussianCurvature(WeightedLoss):
    _req_hess = True

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        mat = torch.cat([hess, grad[..., None]], -1)
        row = torch.cat([grad, torch.zeros_like(grad[..., 0])[..., None]], -1)
        mat = torch.cat([mat, row[..., None, :]], -2)
        determinants = torch.linalg.det(mat)
        norm_4 = (grad**2).sum(dim=-1)**2
        return - determinants / norm_4


class HessianAlign(WeightedLoss):
    _req_grad = True
    _req_hess = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = FloatParameter('delta', 10., 0.1, 100)

    def energy(self, x, y, pred, y_grad=None, grad=None, hess=None):
        grad = torch.nn.functional.normalize(grad, dim=-1)
        energy = torch.matmul(hess, grad.unsqueeze(-1)).squeeze().square().sum(dim=-1)

        weights = torch.exp(-self.delta.value * torch.abs(pred))
        return energy * weights
