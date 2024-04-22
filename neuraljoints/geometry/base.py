from neuraljoints.utils.parameters import Parameter
from neuraljoints.utils.utils import RegisteredMeta


class Entity(metaclass=RegisteredMeta):
    __names = []

    def __init__(self, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
            name_ = name
            count = 2
            while name_ in Entity.__names:
                name_ = f'{name} {count}'
                count += 1
            name = name_
        if name not in Entity.__names:
            Entity.__names.append(name)
        self.name = name
        super().__init__(**kwargs)

    @property
    def hparams(self):
        return [v for v in self.__dict__.values() if isinstance(v, Parameter)]

    @property
    def entities(self):
        return [v for v in self.__dict__.values() if isinstance(v, Entity)]


class Set(Entity):
    def __init__(self, children=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = set() if children is None else set(children)

    def add(self, child):
        self.children.add(child)

    def remove(self, child):
        self.children.remove(child)

    def empty(self):
        self.children = set()

    def foreach(self, func):
        return [func(c) for c in self.children]

    def __contains__(self, child) -> bool:
        return child in self.children
