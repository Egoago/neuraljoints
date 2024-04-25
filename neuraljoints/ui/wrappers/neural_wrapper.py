import cumcubes
import numpy as np
import torch
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient
from neuraljoints.neural.losses import CompositeLoss, Loss
from neuraljoints.neural.model import Network, Layer
from neuraljoints.neural.sampling import Sampler
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, get_wrapper
from neuraljoints.ui.wrappers.implicit_wrapper import IMPLICIT_PLANE
from neuraljoints.utils.parameters import Parameter, BoolParameter, FloatParameter


class NetworkWrapper(EntityWrapper):
    TYPE = Network

    class ImplicitNetwork(Implicit):
        def __init__(self, model: Network, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        @property
        def hparams(self):
            def params(obj) -> list[Parameter]:
                return [v for v in obj.__dict__.values() if isinstance(v, Parameter)]

            layer_params = []
            for layer in self.model.layers:
                layer_params += params(layer)

            return params(self.model) + layer_params

        def forward(self, position):
            with torch.no_grad():
                position = torch.tensor(position.astype(np.float32), device=self.device)
                values = self.model(position).cpu().numpy()
            return values

        def gradient(self, position):
            raise NotImplementedError()

    def __init__(self, network: Network):
        super().__init__(object=NetworkWrapper.ImplicitNetwork(network))
        self.render_surface = BoolParameter('render surface', False)
        self.smooth = BoolParameter('smooth', False)
        self.isosurface = FloatParameter('isosurface', 0, -1, 1)

    @property
    def network(self) -> ImplicitNetwork:
        return self.object

    def draw_ui(self) -> bool:
        changed, select = imgui.Checkbox('', self.network.name == IMPLICIT_PLANE.selected)
        if select:
            IMPLICIT_PLANE.select(self.object.name)
        elif self.network.name == IMPLICIT_PLANE.selected:
            IMPLICIT_PLANE.select(None)
        imgui.SameLine()
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
        self.changed |= changed
        return self.changed

    def draw_geometry(self):
        IMPLICIT_PLANE.add_scalar_texture(self.network.name, self.network)

        if self.render_surface.value:
            bounds = torch.tensor(IMPLICIT_PLANE.bounds.value, device=self.network.device, dtype=torch.float32)
            res = IMPLICIT_PLANE.resolution.value // 2
            with torch.no_grad():
                X, Y, Z = torch.meshgrid(torch.linspace(-bounds[0], bounds[0], res, device='cuda', dtype=torch.float32),
                                         torch.linspace(-bounds[1], bounds[1], res, device='cuda', dtype=torch.float32),
                                         torch.linspace(-bounds[2], bounds[2], res, device='cuda', dtype=torch.float32),
                                         indexing="ij")
                position = torch.stack([X, Y, Z], dim=-1)

                values = self.network.model(position)

                vertices, faces = cumcubes.marching_cubes(values, self.isosurface.value)
                vertices = vertices / res * 2 * bounds - bounds + bounds/res

                sm = ps.register_surface_mesh("Surface", vertices.cpu().numpy(), faces.cpu().numpy(),
                                              smooth_shade=self.smooth.value)
            if IMPLICIT_PLANE.gradient.value:
                vertices.requires_grad = True
                plane_points = IMPLICIT_PLANE.points[::4, ::4].reshape(-1, 3)
                plane_points = torch.tensor(plane_points.astype(np.float32), device=vertices.device)
                positions = torch.cat([plane_points, vertices])

                pred = self.network.model(positions)
                gradients = gradient(pred, positions)

                gradients = gradients.detach().cpu().numpy()
                plane_grad = gradients[:len(plane_points)]
                surface_grad = gradients[len(plane_points):]

                IMPLICIT_PLANE.add_vector_field(self.network.name, values=plane_grad)
                sm.add_vector_quantity('Normals', surface_grad*0.1, vectortype='ambient', defined_on='vertices',
                                       radius=0.01, color=(0.1, 0.1, 0.1), enabled=True)
        else:
            ps.remove_surface_mesh("Surface", False)


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
            ps.register_point_cloud('sampled points', self.sampler.prev_x)
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
        if imgui.Button('render') or (self.trainer.training and self.trainer.step % 10 == 0):
            self.changed = True

        imgui.SameLine()
        if imgui.Button('rebuild'):
            self.trainer.reset()
            self.trainer.sampler.prev_y = None
            self.changed = True

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
