import time

import numpy as np
import torch
from polyscope import imgui

from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers import EntityWrapper, ImplicitWrapper


class Model2Implicit(Implicit):
    def __init__(self, model: Network, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device

    def forward(self, position):
        with torch.no_grad():
            position = torch.tensor(position.astype(np.float32), device=self.device)
            values = self.model(position).cpu().numpy()
        return values

    def gradient(self, position):
        raise NotImplementedError()


class TrainerWrapper(EntityWrapper):
    def __init__(self, trainer: Trainer, **kwargs):
        super().__init__(entity=trainer, **kwargs)
        self.last_draw = None

    @property
    def trainer(self) -> Trainer:
        return self.entity

    def draw_ui(self):
        super().draw_ui()
        imgui.Begin('Training', True, imgui.ImGuiWindowFlags_AlwaysAutoResize)

        imgui.BeginDisabled(self.trainer.training)
        if imgui.Button('start'):
            self.trainer.train()
        imgui.EndDisabled()

        imgui.SameLine()
        imgui.BeginDisabled(not self.trainer.training)
        if imgui.Button('stop'):
            self.trainer.stop()
        imgui.EndDisabled()

        imgui.SameLine()
        if (imgui.Button('render') or
           (self.trainer.training and self.trainer.step % 10 == 0)):
            self.render()

        imgui.SameLine()
        if imgui.Button('reset'):
            self.trainer.reset()
            self.render()

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

    def render(self):
        implicit_model = Model2Implicit(self.trainer.model, self.trainer.device)
        ImplicitWrapper.draw_implicit(implicit_model)
