from abc import abstractmethod

import torch

from neuraljoints.geometry.base import List
from neuraljoints.geometry.implicit import Implicit, Offset
from neuraljoints.utils.parameters import FloatParameter


class Aggregate(Implicit, List):
    def forward(self, position):
        if len(self.children) == 0:
            return torch.zeros_like(position[..., 0])
        return self.reduce(self.foreach(lambda child: child(position)))

    @abstractmethod
    def reduce(self, values: [torch.Tensor]):
        pass


class Union(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).amin(dim=0)


class InverseUnion(Union):
    def reduce(self, values: [torch.Tensor]):
        return -super().reduce(values)


class Sum(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).sum(dim=0)


class CleanUnion(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        values = torch.stack(values, dim=-1)
        return values.sum(dim=-1) - ((values**2).sum(dim=-1)).sqrt()


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


class Intersect(Aggregate):
    def reduce(self, values: [torch.Tensor]):
        return torch.stack(values, dim=0).amax(dim=0)


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


class IPatch(Aggregate):
    def __init__(self, *args, children: list[Implicit]=None, **kwargs):
        self.w0 = FloatParameter('w0', 1, min=-1., max=1.)
        if children is not None:
            for i in range(len(self.children)):
                name = f'w{i+1}'
                setattr(self, name, FloatParameter(name, 1, min=-1., max=1.))
            children = children + self.get_boundaries(children)
        super().__init__(*args, children=children, **kwargs)

    def add(self, child):
        n = len(self.children) // 2
        implicits = self.children[:n]
        boundaries = self.children[n:]
        name = f'w{n+1}'
        setattr(self, name, FloatParameter(name, 1, min=-1., max=1.))
        boundary = Offset(child=child)
        boundary.offset.value = 0.1
        implicits.append(child)
        boundaries.append(boundary)
        self.children = implicits + boundaries

    def remove(self, child):
        n = len(self.children) // 2
        implicits = self.children[:n]
        boundaries = self.children[n:]
        i = implicits.index(child)
        implicits.pop(i)
        boundaries.pop(i)
        delattr(self, f'w{n}')
        self.children = implicits + boundaries

    def get_boundaries(self, implicits) -> list[Implicit]:
        boundaries = []
        for implicit in implicits:
            boundary = Offset(child=implicit)
            boundary.offset.value = 0.1
            boundaries.append(boundary)
        return boundaries

    def reduce(self, values: [torch.Tensor]):
        implicits, boundaries = torch.stack(values, dim=0).chunk(2, dim=0)
        boundaries = boundaries ** 2
        b_prod = boundaries.prod(dim=0)
        dividend, divisor = -self.w0.value * b_prod, 0
        for i in range(len(implicits)):
            wi = getattr(self, f'w{i + 1}').value
            temp = wi * b_prod / boundaries[i]
            dividend += temp * implicits[i]
            divisor += temp
        return dividend


class IPatchManual(Aggregate):
    def __init__(self, *args, **kwargs):
        implicits = Union(name='implicits')
        boundaries = Union(name='boundaries')
        self.w0 = FloatParameter('w0', 1, min=-1., max=1.)
        super().__init__(*args, children=[implicits, boundaries], **kwargs)

    def forward(self, position):
        if len(self.children[0].children) == len(self.children[1].children) and len(self.children[0].children) > 0:
            implicits = torch.stack(self.children[0].foreach(lambda child: child(position)), dim=0)
            boundaries = torch.stack(self.children[1].foreach(lambda child: child(position)), dim=0)
            return self.reduce([implicits, boundaries])
        return torch.zeros_like(position[..., 0])

    def reduce(self, values: [torch.Tensor]):
        implicits, boundaries = values
        boundaries = boundaries ** 2
        b_prod = boundaries.prod(dim=0)
        dividend, divisor = -self.w0.value * b_prod, 0
        for i in range(len(implicits)):
            temp = b_prod / boundaries[i]
            dividend += temp * implicits[i]
            divisor += temp
        return dividend

