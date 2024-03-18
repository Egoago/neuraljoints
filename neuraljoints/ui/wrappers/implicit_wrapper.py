import polyscope as ps
from polyscope import imgui

from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.utils.parameters import IntParameter


class ImplicitWrapper(EntityWrapper):
    scalar_args = {'defined_on': 'nodes',
                   'datatype': 'symmetric', 'cmap': 'blue-red',
                   # 'enable_isosurface_viz': True, 'isosurface_color': (0.4, 0.6, 0.6),
                   }
    RESOLUTION = IntParameter('resolution', 100, 2, 500)    #TODO add static params
    BOUND = 2

    bound_low = (-2, -2, -0.02)
    bound_high = (2, 2, 0)
    _grid = None

    def __init__(self, implicit: Implicit, **kwargs):
        super().__init__(entity=implicit, **kwargs)

    @classmethod
    @property
    def grid(cls):
        if ImplicitWrapper._grid is None:
            dims = (ImplicitWrapper.RESOLUTION.value, ImplicitWrapper.RESOLUTION.value, 2)
            bound_low = (-ImplicitWrapper.BOUND, -ImplicitWrapper.BOUND,
                         -ImplicitWrapper.BOUND / ImplicitWrapper.RESOLUTION.value*2)
            bound_high = (ImplicitWrapper.BOUND, ImplicitWrapper.BOUND, 0)
            ImplicitWrapper._grid = ps.register_volume_grid("Grid", dims, bound_low, bound_high)
            ImplicitWrapper._grid.set_cull_whole_elements(False)
        return ImplicitWrapper._grid

    @property
    def implicit(self) -> Implicit:
        return self.entity

    def draw_geometry(self):
        super().draw_geometry()
        self.draw_implicit(self.implicit)

    @classmethod
    def draw_implicit(cls, implicit: Implicit):
        cls.grid.add_scalar_quantity_from_callable(implicit.name, implicit, isolines_enabled=True, **cls.scalar_args)


class AggregateWrapper(ImplicitWrapper):    #TODO move to drawable
    def __init__(self, implicit: Aggregate, children: [ImplicitWrapper], **kwargs):
        super().__init__(implicit, **kwargs)
        self.children = children

    def draw_ui(self):
        super().draw_ui()

        if imgui.TreeNode('children'):
            for child in self.children:
                child.draw_ui()
                self.changed = child.changed or self.changed
            imgui.TreePop()

    def draw_geometry(self):
        super().draw_geometry()
        for child in self.children:
            child.draw_geometry()
