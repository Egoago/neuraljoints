from abc import abstractmethod

import torch

from neuraljoints.geometry.base import List
from neuraljoints.geometry.implicit import Implicit, TransformedImplicit
from neuraljoints.utils.parameters import FloatParameter, BoolParameter


class Aggregate(List, Implicit):
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
        self.a0 = FloatParameter(name='a0', initial=1, min=0., max=100.)
        super().__init__(*args, **kwargs)
        for i in range(len(self.children)):
            name = f'a{i+1}'
            setattr(self, name, FloatParameter(name=name, initial=1, min=0., max=20.))

    def reduce(self, values: [torch.Tensor]):
        values = torch.stack(values, dim=-1)
        clean_union = values.sum(dim=-1) - ((values**2).sum(dim=-1)).sqrt()
        a = torch.array([getattr(self, f'a{i + 1}').value for i in range(len(self.children))])
        return clean_union + self.a0.value / (1 + ((values/a)**2).sum(dim=-1))


class RoundUnion(Union):
    def __init__(self, radius=1.0, *args, **kwargs):
        self.radius = FloatParameter(name='radius', initial=radius, min=0., max=2.)
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


class IPatchManual(Aggregate):
    def __init__(self, *args, **kwargs):
        self.boundaries = Union(name='boundaries')
        self.w0 = FloatParameter(name='w0', initial=1, min=-1., max=10.)
        self.filter = BoolParameter(name='filter', initial=True)
        self.exp = FloatParameter(name='exp', initial=2., min=0., max=10.)
        super().__init__(*args, children=[self.boundaries], **kwargs)

    @property
    def implicits(self):
        return self.children[1:]

    def forward(self, position):
        if len(self.implicits) == len(self.boundaries.children) and len(self.implicits) > 0:
            implicits = torch.stack([i(position) for i in self.implicits])
            boundaries = torch.stack([b(position) for b in self.boundaries.children])
            return self.reduce([implicits, boundaries])
        return torch.zeros_like(position[..., 0])

    def reduce(self, values: [torch.Tensor]):
        implicits, boundaries = values
        blended = self.blend(implicits, boundaries)
        if self.filter:
            return torch.where((boundaries > 0).all(dim=0), blended, implicits.min(dim=0).values)
        return blended

    def blend(self, implicits, boundaries):
        boundaries = boundaries ** self.exp.value
        b_prod = boundaries.prod(dim=0)
        dividend, divisor = -self.w0.value * b_prod, 0
        for i in range(len(implicits)):
            temp = b_prod / boundaries[i]
            dividend += temp * implicits[i]
            divisor += temp
        return dividend / divisor


class IPatch(IPatchManual):
    def __init__(self, *args, children: list[Implicit]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = FloatParameter(name='offset', initial=-0.5, min=-1, max=0.)
        if children is not None:
            for child in children:
                self.add(child)

    def add(self, child):
        self.children.append(child)
        self.calculate_boundaries()

    def remove(self, child):
        self.children.remove(child)
        self.calculate_boundaries()

    def calculate_boundaries(self):
        boundaries = []
        if len(self.implicits) > 1:
            for i, implicit in enumerate(self.implicits):
                boundary = TransformedImplicit(child=implicit)
                boundary.offset = self.offset
                boundary.scale.value = -1.0
                boundaries.append(boundary)
            every_other_boundaries = []
            for i, implicit in enumerate(self.implicits):
                every_other_boundaries.append(Union(children=boundaries[:i] + boundaries[i+1:], name=f'{implicit.name} boundary'))
            boundaries = every_other_boundaries
        self.boundaries.children = boundaries


class IPatchHierarchical(Aggregate):
    class IPatchPair(Aggregate):

        def __init__(self, implicit_a, implicit_b, offset, exp, w0, **kwargs):
            super().__init__(children=[implicit_a, implicit_b], **kwargs)
            self.exp = exp
            self.w0 = w0
            self.offset = offset
            self.boundaries = [self.get_boundary(implicit) for implicit in self.children[::-1]]

        def get_boundary(self, implicit):
            boundary = TransformedImplicit(child=implicit)
            boundary.offset = self.offset
            boundary.scale.value = -1.0
            return boundary

        def forward(self, position):
            assert len(self.children) == 2
            assert len(self.boundaries) == 2
            implicits = torch.stack([i(position) for i in self.children])
            boundaries = torch.stack([b(position) for b in self.boundaries])
            return self.reduce([implicits, boundaries])

        def reduce(self, values: [torch.Tensor]):
            implicits, boundaries = values
            blended = self.blend(implicits, boundaries)

            return torch.where((boundaries > 0).all(dim=0), blended, implicits.min(dim=0).values)

        def blend(self, implicits, boundaries):
            boundaries = boundaries ** self.exp.value
            b_prod = boundaries.prod(dim=0)
            dividend, divisor = -self.w0.value * b_prod, 0
            for i in range(len(implicits)):
                temp = b_prod / boundaries[i]
                dividend += temp * implicits[i]
                divisor += temp
            return dividend / divisor

    def __init__(self, *args, children: list[Implicit]=None, **kwargs):
        self.w0 = FloatParameter(name='w0', initial=1, min=-1., max=10.)
        self.exp = FloatParameter(name='exp', initial=2., min=0., max=10.)
        self.tree = None
        self.offset = FloatParameter(name='offset', initial=-0.5, min=-1, max=0.)
        super().__init__(*args, children=children, **kwargs)

    def add(self, child):
        self.children.append(child)
        self.build_tree()

    def remove(self, child):
        self.children.remove(child)
        self.build_tree()

    def forward(self, position):
        if self.tree is not None:
            return self.tree(position)
        return torch.zeros_like(position[..., 0])

    def reduce(self, values: [torch.Tensor]):
        raise RuntimeError('Does not use reduce.')

    def build_tree(self):
        self.tree = None
        if len(self.children) > 0:
            self.tree = self.children[0]
            for implicit in self.children[1:]:
                self.tree = self.IPatchPair(implicit_a=self.tree, implicit_b=implicit,
                                            offset=self.offset, exp=self.exp, w0=self.w0)


class Productum(IPatch):
    def blend(self, implicits, boundaries):
        boundaries = boundaries ** self.exp.value
        return boundaries.prod(dim=0)
