import torch
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient, hessian
from neuraljoints.neural.losses import CompositeLoss, Loss
from neuraljoints.neural.model import Network, Layer
from neuraljoints.neural.sampling import Sampler
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, ListWrapper, get_wrapper
from neuraljoints.ui.wrappers.implicit_wrapper import ImplicitWrapper
from neuraljoints.ui.wrappers.implicit_plane import IMPLICIT_PLANE
from neuraljoints.utils.parameters import BoolParameter


class NetworkWrapper(ImplicitWrapper):
    TYPE = Network

    def __init__(self, network: Network):
        super().__init__(object=network)

    @property
    def network(self) -> Network:
        return self.object

    def draw_ui(self) -> bool:
        super().draw_ui()
        # Handle model type change
        init_schemes = Layer.get_subclass(self.network.layer.value).init_schemes
        if set(init_schemes) != set(self.network.init_scheme.choices):
            self.network.init_scheme.choices = init_schemes
            self.network.init_scheme.initial = init_schemes[0]
            self.network.init_scheme.value = init_schemes[0]
        if self.network.layer.value == 'Siren':
            init_scheme = self.network.init_scheme.value
            if init_scheme == 'mfgi':
                self.network.n_layers.min = 3
                self.network.n_layers.value = max(self.network.n_layers.value, 3)
                self.network.n_layers.initial = max(self.network.n_layers.initial, 3)
            elif init_scheme == 'geometric':
                self.network.n_layers.min = 2
                self.network.n_layers.value = max(self.network.n_layers.value, 2)
                self.network.n_layers.initial = max(self.network.n_layers.initial, 2)
            else:
                self.network.n_layers.min = 0
        return self.changed


class SamplerWrapper(EntityWrapper):
    TYPE = Sampler

    def __init__(self, sampler: Sampler):
        super().__init__(object=sampler)
        self.grad = None
        self.render_points = BoolParameter(name='render points', initial=False)

    @property
    def sampler(self) -> Sampler:
        return self.object

    def draw_geometry(self):
        with torch.no_grad():
            if self.sampler.prev_x is not None and self.render_points.value:
                pc = ps.register_point_cloud('sampled points', self.sampler.prev_x.detach().cpu().numpy())
                surface_indices = self.sampler.surface_indices
                if len(surface_indices) > 0:
                    mask = torch.zeros(len(self.sampler.prev_x), device='cpu', dtype=torch.bool)
                    mask[surface_indices] = True
                    pc.add_scalar_quantity('type', mask.numpy(), enabled=True)
            else:
                ps.remove_point_cloud('sampled points', False)


class LossWrapper(ImplicitWrapper):
    TYPE = Loss

    class ImplicitLoss(Implicit):    # TODO refactor
        def __init__(self, loss: Loss, model: Network, target: Implicit, **kwargs):
            super().__init__(**kwargs)
            self.name = loss.name
            self.loss = loss
            self.model = model
            self.target = target

        @property
        def hparams(self):
            return self.loss.hparams

        def forward(self, position):
            outputs = {'x': position}
            if not position.requires_grad and self.loss.req_grad:
                position.requires_grad = True

            outputs['y_gt'] = self.target(position)

            outputs['y_pred'] = self.model(position)

            if 'grad_gt' in self.loss.attributes:
                outputs['grad_gt'] = gradient(outputs['y_gt'], position)
            if 'grad_pred' in self.loss.attributes:
                outputs['grad_pred'] = gradient(outputs['y_pred'], position)
            if 'hess_pred' in self.loss.attributes:
                outputs['hess_pred'] = hessian(outputs['grad_pred'], position)

            return self.loss.energy(**outputs)

    def __init__(self, loss: Loss, trainer: Trainer):
        super().__init__(object=LossWrapper.ImplicitLoss(loss, trainer.model, trainer.implicit))
        self.scalar_args = {'cmap': 'viridis'}
        self.model = trainer.model

    @property
    def surface_implicit(self) -> Implicit:
        return self.model


class CompositeLossWrapper(ListWrapper):
    TYPE = CompositeLoss

    def remove_wrapper(self, wrapper: EntityWrapper) -> bool:
        if wrapper is not None and wrapper in self.child_wrappers:
            self.child_wrappers.remove(wrapper)
            self.object.remove(wrapper.object.loss)
            self.changed = True
            return True
        return False

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
        self.loss_wrapper = get_wrapper(trainer.loss_fn, trainer=trainer)
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
        if imgui.Button('render') or (self.trainer.training and self.trainer.training_step % 5 == 0):
            self.changed = True
            self.network_wrapper.changed = True
            IMPLICIT_PLANE.startup = True

        imgui.SameLine()
        if imgui.Button('rebuild'):
            self.trainer.reset()
            self.changed = True
            self.network_wrapper.changed = True
            self.loss_wrapper.changed = True

        if self.trainer.training:
            ratio = self.trainer.training_step / self.trainer.max_steps.value
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
