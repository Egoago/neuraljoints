from abc import abstractmethod, ABC

import torch

from neuraljoints.geometry.base import Set
from neuraljoints.neural.autograd import gaussian_curvature
from neuraljoints.neural.sampling import Pipeline
from neuraljoints.utils.parameters import FloatParameter


class Loss(Pipeline):
    _on_surface = False
    _on_volume = False

    @property
    def enabled(self):
        return True

    @property
    def attributes(cls) -> set[str]:
        return super().attributes | {'x'}

    def __call__(self, **kwargs):
        return self.energy(**kwargs).mean()

    def energy(self, **kwargs):
        self.check_input(**kwargs)
        x = kwargs['x']

        if (self._on_surface or self._on_volume) and 'surface_indices' in kwargs:
            index = kwargs['surface_indices']
            if self._on_volume:
                index = torch.ones(len(x), dtype=torch.bool, device=x.device)
                index[kwargs['surface_indices']] = False
            kwargs = {k: v[index] for k, v in kwargs.items() if k != 'surface_indices'}

        if self.enabled:
            return self._energy(**kwargs)
        return torch.zeros_like(x[..., 0])

    @abstractmethod
    def _energy(self, **kwargs):
        raise NotImplementedError()


class WeightedLoss(Loss, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = FloatParameter('weight', 1, 0, 1)

    def __call__(self, **kwargs):
        return Loss.__call__(self, **kwargs) * (self.weight.value * 1000)


class CompositeLoss(Loss, Set):
    @property
    def attributes(self) -> set[str]:
        attr = super().attributes
        for loss in self.children:
            attr |= loss.attributes
        return attr

    @property
    def losses(self) -> set[Loss]:
        return self.children

    def _energy(self, **kwargs):
        sum = 0.
        for loss in self.children:
            sum = sum + loss(**kwargs)
        return sum


class Mse(WeightedLoss):
    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'y_pred', 'y_gt'}

    def _energy(self, y_gt, y_pred, **kwargs):
        return (y_gt - y_pred) ** 2


class Eikonal(WeightedLoss):
    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'grad_pred'}

    def _energy(self, grad_pred, **kwargs):
        grad_norm = torch.linalg.norm(grad_pred, axis=-1)
        return (grad_norm - 1).abs()


class EikonalRelaxed(WeightedLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_threshold = FloatParameter('grad threshold', 0.8, 0, 1.)
        self.relu = torch.nn.ReLU()

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'grad_pred'}

    def _energy(self, grad_pred, **kwargs):
        grad_norm = torch.linalg.norm(grad_pred, axis=-1)
        return self.relu(self.grad_threshold.value - grad_norm)


class Dirichlet(WeightedLoss):
    _on_surface = True

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'grad_pred'}

    def _energy(self, grad_pred, **kwargs):
        return (grad_pred ** 2).sum(dim=-1)


class Neumann(WeightedLoss):
    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'grad_pred', 'grad_gt'}

    def _energy(self, grad_gt, grad_pred, **kwargs):
        grad_pred = torch.nn.functional.normalize(grad_pred, dim=-1)
        return 1 - (grad_gt * grad_pred).sum(dim=-1)


class Laplacian(WeightedLoss):
    _on_surface = True

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'hess_pred'}

    def _energy(self, hess_pred, **kwargs):
        trace = torch.einsum("...ii", hess_pred)  # trace of the hessian is the laplacian

        return trace


class HessianDet(WeightedLoss):
    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'hess_pred'}

    def _energy(self, hess_pred, **kwargs):
        determinants = torch.linalg.det(hess_pred)
        return torch.abs(determinants)


class GaussianCurvature(WeightedLoss):
    _on_surface = True

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'hess_pred', 'grad_pred'}

    def _energy(self, hess_pred, grad_pred, **kwargs):
        curvature = gaussian_curvature(hess_pred, grad_pred)
        return curvature


class DoubleThrough(GaussianCurvature):
    @staticmethod
    def double_through(x):
        pi = torch.pi
        return (64 * pi - 80) / (pi ** 4) * (x ** 4) - (64 * pi - 88) / (pi ** 3) * (x ** 3) + (16 * pi - 29) / (
                    pi ** 2) * (x ** 2) + 3 * x / pi

    def _energy(self, **kwargs):
        energy = super()._energy(**kwargs)
        return DoubleThrough.double_through(energy)


class HessianAlign(WeightedLoss):
    _on_surface = True

    @property
    def attributes(self) -> set[str]:
        return super().attributes | {'hess_pred', 'grad_pred', 'y_pred'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = FloatParameter('delta', 10., 0.1, 100)

    def _energy(self, y_pred, grad_pred, hess_pred, **kwargs):
        grad = torch.nn.functional.normalize(grad_pred, dim=-1)
        energy = torch.matmul(hess_pred, grad.unsqueeze(-1)).squeeze().square().sum(dim=-1)

        weights = torch.exp(-self.delta.value * torch.abs(y_pred))
        return energy * weights
