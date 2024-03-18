from neuraljoints.geometry.aggregate import Aggregate
from neuraljoints.geometry.base import Entity, ControlPoints
from neuraljoints.geometry.implicit import Implicit
from neuraljoints.geometry.parametric import Parametric
from neuraljoints.neural.trainer import Trainer
from neuraljoints.ui.wrappers.control_point_wrapper import ControlPointsWrapper
from neuraljoints.ui.wrappers.entity_wrapper import EntityWrapper
from neuraljoints.ui.wrappers.implicit_wrapper import AggregateWrapper, \
    ImplicitWrapper
from neuraljoints.ui.wrappers.parametric_wrapper import ParametricWrapper
from neuraljoints.ui.wrappers.trainer_wrapper import TrainerWrapper


def get_wrapper(entity: Entity):
    entity_wrappers = [get_wrapper(e) for e in entity.entities]
    if isinstance(entity, Aggregate):
        children = [get_wrapper(e) for e in entity.children]
        return AggregateWrapper(entity, children=children, entity_wrappers=entity_wrappers)
    if isinstance(entity, ControlPoints):
        return ControlPointsWrapper(entity, entity_wrappers=entity_wrappers)
    if isinstance(entity, Parametric):
        return ParametricWrapper(entity, entity_wrappers=entity_wrappers)
    if isinstance(entity, Implicit):
        return ImplicitWrapper(entity, entity_wrappers=entity_wrappers)
    if isinstance(entity, Trainer):
        return TrainerWrapper(entity, entity_wrappers=entity_wrappers)
    return EntityWrapper(entity, entity_wrappers=entity_wrappers)
