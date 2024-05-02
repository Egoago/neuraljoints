from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient
from neuraljoints.neural.losses import CompositeLoss, Loss
from neuraljoints.neural.model import Network, Layer
from neuraljoints.neural.sampling import Sampler
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, get_wrapper
from neuraljoints.ui.wrappers.implicit_wrapper import ImplicitWrapper
from neuraljoints.utils.parameters import Parameter, BoolParameter


class NetworkWrapper(ImplicitWrapper):
    TYPE = Network

    class ImplicitNetwork(Implicit):    # TODO refactor
        def __init__(self, model: Network, **kwargs):
            super().__init__(**kwargs)
            self.model = model

        @property
        def hparams(self):
            def params(obj) -> list[Parameter]:
                return [v for v in obj.__dict__.values() if isinstance(v, Parameter)]

            layer_params = []
            for layer in self.model.layers:
                layer_params += params(layer)

            return params(self.model) + layer_params

        def forward(self, position):
            return self.model(position)

        def gradient(self, position):   # TODO refactor
            position.requires_grad = True
            pred = self.model(position)
            return gradient(pred, position)

    def __init__(self, network: Network):
        super().__init__(object=NetworkWrapper.ImplicitNetwork(network))

    @property
    def network(self) -> ImplicitNetwork:
        return self.object

    def draw_ui(self) -> bool:
        super().draw_ui()
        # Handle model type change
        init_schemes = Layer.get_subclass(self.network.model.layer.value).init_schemes
        if set(init_schemes) != set(self.network.model.init_scheme.choices):
            self.network.model.init_scheme.choices = init_schemes
            self.network.model.init_scheme.initial = init_schemes[0]
            self.network.model.init_scheme.value = init_schemes[0]
        if self.network.model.layer.value == 'Siren':
            init_scheme = self.network.model.init_scheme.value
            if init_scheme == 'mfgi':
                self.network.model.n_layers.min = 3
                self.network.model.n_layers.value = max(self.network.model.n_layers.value, 3)
                self.network.model.n_layers.initial = max(self.network.model.n_layers.initial, 3)
            elif init_scheme == 'geometric':
                self.network.model.n_layers.min = 2
                self.network.model.n_layers.value = max(self.network.model.n_layers.value, 2)
                self.network.model.n_layers.initial = max(self.network.model.n_layers.initial, 2)
            else:
                self.network.model.n_layers.min = 0
        return self.changed


class SamplerWrapper(EntityWrapper):
    TYPE = Sampler

    def __init__(self, sampler: Sampler):
        super().__init__(object=sampler)
        self.grad = None
        self.render_points = BoolParameter('render points', False)

    @property
    def sampler(self) -> Sampler:
        return self.object

    def draw_geometry(self):
        if self.sampler.prev_x is not None and self.render_points.value:
            ps.register_point_cloud('sampled points', self.sampler.prev_x.detach().cpu().numpy())
        else:
            ps.remove_point_cloud('sampled points', False)


class CompositeLossWrapper(SetWrapper):
    TYPE = CompositeLoss

    @property
    def choices(self) -> set[Loss] | None:
        composite_loss: CompositeLoss = self.object
        used_classes = {loss.__class__ for loss in composite_loss.losses}
        return {i for i in Loss.subclasses if i not in used_classes}


class TrainerWrapper(EntityWrapper):
    TYPE = Trainer

    def __init__(self, trainer: Trainer):
        super().__init__(object=trainer)
        self.last_draw = None
        self.loss_wrapper = get_wrapper(trainer.loss_fn)
        self.sampler_wrapper = get_wrapper(trainer.sampler)
        self.network_wrapper = get_wrapper(trainer.model)

    @property
    def trainer(self) -> Trainer:
        return self.object

    def draw_ui(self) -> bool:
        imgui.Begin('Training', True, imgui.ImGuiWindowFlags_AlwaysAutoResize)
        super().draw_ui()
        self.changed = False

        self.network_wrapper.draw_ui()
        self.sampler_wrapper.draw_ui()
        self.loss_wrapper.draw_ui()

        if self.trainer.training:
            if imgui.Button('stop'):
                self.trainer.stop()
        else:
            if imgui.Button('start'):
                self.trainer.train()

        imgui.SameLine()
        if imgui.Button('render') or (self.trainer.training and self.trainer.step % 5 == 0):
            self.changed = True
            self.network_wrapper.changed = True

        imgui.SameLine()
        if imgui.Button('rebuild'):
            self.trainer.reset()
            self.trainer.sampler.prev_y = None
            self.changed = True
            self.network_wrapper.changed = True

        if self.trainer.training:
            ratio = self.trainer.step / self.trainer.max_steps.value
            imgui.ProgressBar(ratio, (-1, 0))

        if len(self.trainer.losses) > 0:
            imgui.SetNextItemWidth(-1)
            imgui.PlotLines('', self.trainer.losses, graph_size=(0, 80))
            imgui.Text(f'Loss {self.trainer.losses[-1]:9.2e}')

        imgui.End()
        return self.changed

    def draw_geometry(self):
        super().draw_geometry()
        self.sampler_wrapper.draw_geometry()
        self.network_wrapper.draw_geometry()
