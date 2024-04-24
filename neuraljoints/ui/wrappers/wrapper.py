from neuraljoints.ui.ui import Drawable
from neuraljoints.utils.parameters import Parameter
from neuraljoints.utils.utils import RegisteredMeta


class Wrapper(Drawable, metaclass=RegisteredMeta):
    @property
    def hparams(self):  # parameters name would be better but collides with torch.nn.Module.parameters
        return [v for v in self.__dict__.values() if isinstance(v, Parameter)]
