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

    def __init__(self, implicit: Implicit):
        super().__init__(entity=implicit)
        self.startup = True

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

    def draw(self) -> bool:
        changed = super().draw()
        if changed or self.startup:
            self.grid.add_scalar_quantity_from_callable(self.implicit.name, self.implicit,
                                                        isolines_enabled=True, **self.scalar_args)
        self.startup = False
        return changed


class AggregateWrapper(ImplicitWrapper):
    def __init__(self, implicit: Aggregate):
        super().__init__(implicit)
        self.children = [ImplicitWrapper(c) for c in implicit.children]

    def draw(self) -> bool:
        changed = EntityWrapper.draw(self)

        if imgui.TreeNode('children'):
            for child in self.children:
                changed = child.draw() or changed
            imgui.TreePop()

        if changed or self.startup:
            self.grid.add_scalar_quantity_from_callable(self.implicit.name, self.implicit,
                                                        isolines_enabled=True, **self.scalar_args)
        self.startup = False
        return changed


class ParametricToImplicitBruteWrapper(ImplicitWrapper):
    def __init__(self, implicit: ParametricToImplicitBrute):
        super().__init__(implicit)
        self.parametric_wrapper = ParametricWrapper(implicit.parametric)

    def draw(self) -> bool:
        changed = EntityWrapper.draw(self)
        changed = self.parametric_wrapper.draw() or changed

        if changed or self.startup:
            self.grid.add_scalar_quantity_from_callable(self.implicit.name, self.implicit,
                                                        isolines_enabled=True, **self.scalar_args)
        self.startup = False
        return changed
