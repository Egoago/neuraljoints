from abc import abstractmethod

import torch

from neuraljoints.geometry.base import Set
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.utils.parameters import FloatParameter


class Aggregate(Implicit, Set):
    def forward(self, position):
        if len(self.children) == 0:
            return torch.zeros_like(position[..., 0])
        return self.reduce(self.foreach(lambda child: child(position)))

    def gradient(self, position):
        if len(self.children) == 0:
            return torch.zeros_like(position)
        else:
            values, gradients = list(zip(*self.foreach(lambda child: child(position, grad=True))))
            return self.reduce_gradient(values, gradients)

    @abstractmethod
    def reduce(self, values: [torch.Tensor]):
        pass

    @abstractmethod
    def reduce_gradient(self, values: [torch.Tensor], gradients: [torch.Tensor]):
        pass


class Union(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).amin(dim=0)

    def reduce_gradient(self, values, gradients):
        values = torch.stack(values, dim=-1)
        gradients = torch.stack(gradients, dim=-2)
        indices = torch.argmin(values, dim=-1)
        return torch.take_along_dim(gradients, indices[..., None, None], dim=-2).squeeze()


class InverseUnion(Union):
    def reduce(self, values: [torch.Tensor]):
        return -super().reduce(values)

    def reduce_gradient(self, values, gradients):
        return -super().reduce_gradient(values, gradients)


class Sum(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).sum(dim=0)

    def reduce_gradient(self, values, gradients):
        return torch.stack(gradients, dim=0).sum(dim=0)


class CleanUnion(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        values = torch.stack(values, dim=-1)
        return values.sum(dim=-1) - ((values**2).sum(dim=-1)).sqrt()

    def reduce_gradient(self, values, gradients):
        values = torch.stack(values, dim=-1)
        gradients = torch.stack(gradients, dim=-1)
        return gradients.sum(dim=-1) - ((values[..., None, :] * gradients).sum(dim=-1)/((values**2).sum(dim=-1)).sqrt()[..., None])


class RUnion(Aggregate):
    def __init__(self, *args, **kwargs):
        self.a0 = FloatParameter('a0', 1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'a{i+1}'
            setattr(self, name, FloatParameter(name, 1, min=0., max=20.))

    def reduce(self, values: [torch.Tensor]):
        values = torch.stack(values, dim=-1)
        clean_union = values.sum(dim=-1) - ((values**2).sum(dim=-1)).sqrt()
        a = torch.array([getattr(self, f'a{i + 1}').value for i in range(len(self.children))])
        return clean_union + self.a0.value / (1 + ((values/a)**2).sum(dim=-1))

    def reduce_gradient(self, values: [torch.Tensor], gradients: [torch.Tensor]):
        raise NotImplementedError()


class RoundUnion(Union):
    def __init__(self, radius=1.0, *args, **kwargs):
        self.radius = FloatParameter('radius', radius, min=0., max=2.)
        super().__init__(*args, **kwargs)

    def reduce(self, values: [torch.Tensor]):
        #https://www.ronja-tutorials.com/post/035-2d-sdf-combination/
        intersectionSpace = torch.stack(values, dim=-1) - self.radius.value
        intersectionSpace = torch.clamp(intersectionSpace, max=0)
        insideDist = -torch.linalg.norm(intersectionSpace, dim=-1)
        union = super().reduce(values)
        outsideDist = torch.clamp(union, min=self.radius.value)
        return insideDist + outsideDist

    def reduce_gradient(self, values: [torch.Tensor], gradients: [torch.Tensor]):
        raise NotImplementedError()


class Intersect(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).amax(dim=0)

    def reduce_gradient(self, values, gradients):
        values = torch.stack(values, dim=-1)
        gradients = torch.stack(gradients, dim=-2)
        indices = torch.armax(values, dim=-1)
        return torch.take_along_dim(gradients, indices[..., None, None], dim=-2).squeeze()


class SuperElliptic(Aggregate):
    def __init__(self, *args, **kwargs):
        self.t = FloatParameter('t', 1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'r{i+1}'
            setattr(self, name, FloatParameter(name, 1, min=0., max=20.))

    def reduce(self, values: [torch.Tensor]):
        p = torch.stack(values, dim=-1)
        r = torch.tensor([getattr(self, f'r{i + 1}').value for i in range(len(self.children))],
                         dtype=values[0].dtype, device=values[0].device)
        return 1 - (torch.clamp((1 - p) / r, min=0) ** self.t.value).sum(dim=-1)

    def reduce_gradient(self, values: [torch.Tensor], gradients: [torch.Tensor]):
        raise NotImplementedError()
