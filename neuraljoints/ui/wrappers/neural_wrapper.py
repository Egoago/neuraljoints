import time
from inspect import isabstract

import cumcubes
import numpy as np
import torch
from polyscope import imgui
import polyscope as ps

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient
from neuraljoints.neural.losses import CompositeLoss, Loss
from neuraljoints.neural.model import Network
from neuraljoints.neural.sampling import Sampler
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers.base_wrapper import EntityWrapper, SetWrapper, get_wrapper
from neuraljoints.ui.wrappers.implicit_wrapper import ImplicitWrapper
from neuraljoints.utils.parameters import Parameter


class NetworkWrapper(EntityWrapper):
    TYPE = Network

    class ImplicitNetwork(Implicit):
        def __init__(self, model: Network, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        @property
        def hparams(self):
            return [v for v in self.model.__dict__.values() if isinstance(v, Parameter)]

        def forward(self, position):
            with torch.no_grad():
                position = torch.tensor(position.astype(np.float32), device=self.device)
                values = self.model(position).cpu().numpy()
            return values

        def gradient(self, position):
            raise NotImplementedError()

    def __init__(self, network: Network):
        super().__init__(object=NetworkWrapper.ImplicitNetwork(network))

    @property
    def network(self) -> ImplicitNetwork:
        return self.object

    def draw_ui(self) -> bool:
        changed, select = imgui.Checkbox('', self.network.name == ImplicitWrapper._selected)
        if select:
            ImplicitWrapper._selected = self.object.name
        imgui.SameLine()
        super().draw_ui()
        self.changed |= changed
        return self.changed

    def draw_geometry(self):
        ImplicitWrapper.add_scalar_texture(self.network.name, self.network)

        bounds = torch.tensor(ImplicitWrapper.BOUNDS.value, device=self.network.device, dtype=torch.float32)
        res = ImplicitWrapper.RESOLUTION.value // 2
        with torch.no_grad():
            X, Y, Z = torch.meshgrid(torch.linspace(-bounds[0], bounds[0], res, device='cuda', dtype=torch.float32),
                                     torch.linspace(-bounds[1], bounds[1], res, device='cuda', dtype=torch.float32),
                                     torch.linspace(-bounds[2], bounds[2], res, device='cuda', dtype=torch.float32),
                                     indexing="ij")
            position = torch.stack([X, Y, Z], dim=-1)

            values = self.network.model(position)

            vertices, faces = cumcubes.marching_cubes(values, 0)
            vertices = vertices / res * 2 * bounds - bounds + bounds/res

            sm = ps.register_surface_mesh("Surface", vertices.cpu().numpy(), faces.cpu().numpy(),)
        if ImplicitWrapper.GRADIENT.value:
            vertices.requires_grad = True
            plane_points = ImplicitWrapper.points[::2, ::2].reshape(-1, 3)
            plane_points = torch.tensor(plane_points.astype(np.float32), device=vertices.device)
            positions = torch.cat([plane_points, vertices])

            pred = self.network.model(positions)
            gradients = gradient(pred, positions)

            gradients = gradients.detach().cpu().numpy()
            plane_grad = gradients[:len(plane_points)]
            surface_grad = gradients[len(plane_points):]

            ImplicitWrapper.add_vector_field(self.network.name, values=plane_grad)
            sm.add_vector_quantity('Normals', surface_grad*0.1, vectortype='ambient', defined_on='vertices',
                                   radius=0.01, color=(0.1, 0.1, 0.1), enabled=True)


class SamplerWrapper(EntityWrapper):
    TYPE = Sampler

    def __init__(self, sampler: Sampler):
        super().__init__(object=sampler)
        self.grad = None

    @property
    def sampler(self) -> Sampler:
        return self.object

    def draw_geometry(self):
        if self.sampler.prev_x is not None:
            ps.register_point_cloud('sampled points', self.sampler.prev_x)


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
        if imgui.Button('reset'):
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

            nanoseconds = time.time_ns()
            if self.last_draw is not None:
                imgui.SameLine()
                imgui.Text(f'{1e9/(nanoseconds - self.last_draw):.1f}fps')
            self.last_draw = nanoseconds
        imgui.End()
        return self.changed

    def draw_geometry(self):
        super().draw_geometry()
        self.sampler_wrapper.draw_geometry()
        self.network_wrapper.draw_geometry()
