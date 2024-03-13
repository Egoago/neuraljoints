import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.implicit import Implicit, ParametricToImplicitBrute
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.ui.wrappers.parametric_wrapper import ParametricWrapper


class ImplicitWrapper(EntityWrapper):
    scalar_args = {'enabled': True, 'defined_on': 'nodes',
                   'datatype': 'symmetric', 'cmap': 'blue-red',
                   # 'enable_isosurface_viz': True, 'isosurface_color': (0.4, 0.6, 0.6),
                   }
    dims = (200, 200, 2)

    bound_low = (-2, -2, -0.02)
    bound_high = (2, 2, 0)
    _grid = None

    def __init__(self, implicit: Implicit, **kwargs):
        super().__init__(entity=implicit, **kwargs)

    @property
    def grid(self):
        if ImplicitWrapper._grid is None:
            ImplicitWrapper._grid = ps.register_volume_grid("Grid", ImplicitWrapper.dims,
                                                            ImplicitWrapper.bound_low,  # TODO add to parameters
                                                            ImplicitWrapper.bound_high)
            ImplicitWrapper._grid.set_cull_whole_elements(False)
        return ImplicitWrapper._grid

    @property
    def implicit(self) -> Implicit:
        return self.entity

    def draw_geometry(self):
        super().draw_geometry()
        self.grid.add_scalar_quantity_from_callable(self.implicit.name, self.implicit,
                                                    isolines_enabled=True, **self.scalar_args)


class AggregateWrapper(ImplicitWrapper):
    def __init__(self, implicit: Aggregate, children: [ImplicitWrapper], **kwargs):
        super().__init__(implicit)
        self.children = children

    def draw_ui(self):
        super().draw_ui()

        if imgui.TreeNode('children'):
            for child in self.children:
                child.draw_ui()
                self.changed = child.changed or self.changed
            imgui.TreePop()
