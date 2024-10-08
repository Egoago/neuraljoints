﻿import uuid
import random
from abc import abstractmethod

from polyscope_bindings import imgui

from neuraljoints.geometry.base import Entity, List, Proxy
from neuraljoints.ui.wrappers.parameter_wrapper import ParameterWrapper
from neuraljoints.ui.wrappers.wrapper import Wrapper


class EntityWrapper(Wrapper):
    TYPE = Entity

    def __init__(self, object: TYPE, **kwargs):
        super().__init__(**kwargs)
        self.object = object
        self.id = str(uuid.uuid4())
        self.color = [random.random() for _ in range(3)]
        self.type_bar = True

    def draw_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.object.hparams + self.hparams
        for hparam in parameters:
            self.changed |= ParameterWrapper.draw(hparam)

    def draw_ui(self) -> bool:
        super().draw_ui()

        imgui.PushId(self.id)
        if imgui.TreeNode(self.object.name):
            self.draw_parameters()

            imgui.TreePop()
        imgui.PopID()
        return self.changed


class ListWrapper(EntityWrapper):
    TYPE = List

    def __init__(self, object: TYPE, **kwargs):
        super().__init__(object, **kwargs)
        self._changed = False
        self.kwargs = kwargs
        self.child_wrappers: list[EntityWrapper] = object.foreach(get_wrapper, **kwargs)

    @classmethod
    @property
    @abstractmethod
    def choices(cls) -> list[Entity] | None:
        return None

    def remove_wrapper(self, wrapper: EntityWrapper) -> bool:
        if wrapper is not None and wrapper in self.child_wrappers:
            self.object.remove(wrapper.object)
            self.child_wrappers = self.object.foreach(get_wrapper, **self.kwargs)
            self.changed = True
            return True
        return False

    def draw_ui(self) -> bool:
        Wrapper.draw_ui(self)
        imgui.PushId(self.id)
        if imgui.TreeNodeEx(self.object.name, imgui.ImGuiTreeNodeFlags_DefaultOpen):
            self.draw_parameters()

            if self.choices is None:
                for wrapper in self.child_wrappers:
                    self.changed |= wrapper.draw_ui()
            else:  # interactive UI
                object: List = self.object
                child_to_remove = None
                for wrapper in self.child_wrappers:
                    imgui.PushID(wrapper.id + 'node')
                    if imgui.Button(' - '):
                        child_to_remove = wrapper
                    imgui.SameLine()
                    self.changed |= wrapper.draw_ui()
                    imgui.PopID()

                self.remove_wrapper(child_to_remove)

                if imgui.Button(' + '):
                    imgui.OpenPopup(self.id + 'choices')
                if imgui.BeginPopup(self.id + 'choices'):
                    if imgui.BeginTabBar('choices tab bar'):
                        selected, _ = imgui.BeginTabItem('types', True)
                        if not selected:
                            self.type_bar = False
                        if self.type_bar:
                            for choice in self.choices:
                                if imgui.Button(choice.__name__):
                                    new_entity = choice()
                                    object.add(new_entity)
                                    self.child_wrappers = object.foreach(get_wrapper, **self.kwargs)
                                    imgui.CloseCurrentPopup()
                            imgui.EndTabItem()
                        selected, _ = imgui.BeginTabItem('instances', True)
                        if not selected:
                            self.type_bar = True
                        if not self.type_bar:
                            instances = set().union(*[set(cls.instances) for cls in self.choices if hasattr(cls, 'instances')])
                            instances = sorted(instances, key=lambda x: x.name)
                            for instance in instances:
                                if imgui.Button(instance.name):
                                    object.add(instance)
                                    self.child_wrappers = object.foreach(get_wrapper, **self.kwargs)
                                    imgui.CloseCurrentPopup()
                            imgui.EndTabItem()
                        imgui.EndTabBar()
                    imgui.EndPopup()
            imgui.TreePop()
        imgui.PopID()
        return self.changed

    def draw_geometry(self):
        super().draw_geometry()
        for wrapper in self.child_wrappers:
            wrapper.draw_geometry()


class ProxyWrapper(EntityWrapper):
    TYPE = Proxy

    def __init__(self, object: TYPE, **kwargs):
        super().__init__(object, **kwargs)
        self.child_wrapper = None if object.child is None else get_wrapper(object.child)

    @classmethod
    @property
    @abstractmethod
    def choices(cls) -> list[Entity] | None:
        return None

    def empty(self) -> bool:
        if self.child_wrapper is not None:
            self.child_wrapper = None
            self.object.child = None
            self.changed = True
        return True

    def draw_ui(self) -> bool:
        imgui.PushId(self.id)
        if imgui.TreeNodeEx(self.object.name, imgui.ImGuiTreeNodeFlags_DefaultOpen):
            self.draw_parameters()

            if self.choices is None:
                if self.child_wrapper is not None:
                    self.changed |= self.child_wrapper.draw_ui()
            else:  # interactive UI
                if self.child_wrapper is not None:
                    imgui.PushID(self.child_wrapper.id + 'node')
                    remove = imgui.Button(' - ')
                    imgui.SameLine()
                    self.changed |= self.child_wrapper.draw_ui()
                    imgui.PopID()
                    if remove:
                        self.empty()
                else:
                    if imgui.Button(' + '):
                        imgui.OpenPopup(self.id + 'choices')
                    if imgui.BeginPopup(self.id + 'choices'):
                        if imgui.BeginTabBar('choices tab bar'):
                            selected, _ = imgui.BeginTabItem('types', True)
                            if not selected:
                                self.type_bar = False
                            if self.type_bar:
                                for choice in self.choices:
                                    if imgui.Button(choice.__name__):
                                        new_entity = choice()
                                        self.object.child = new_entity
                                        self.child_wrapper = get_wrapper(self.object.child)
                                        imgui.CloseCurrentPopup()
                                imgui.EndTabItem()
                            selected, _ = imgui.BeginTabItem('instances', True)
                            if not selected:
                                self.type_bar = True
                            if not self.type_bar:
                                instances = set().union(
                                    *[set(cls.instances) for cls in self.choices if hasattr(cls, 'instances')])
                                instances = sorted(instances, key=lambda x: x.name)
                                for instance in instances:
                                    if imgui.Button(instance.name):
                                        self.object.child = instance
                                        self.child_wrapper = get_wrapper(self.object.child)
                                        imgui.CloseCurrentPopup()
                                imgui.EndTabItem()
                            imgui.EndTabBar()
                        imgui.EndPopup()
            imgui.TreePop()
        imgui.PopID()
        return self.changed

    def draw_geometry(self):
        super().draw_geometry()
        if self.child_wrapper is not None:
            self.child_wrapper.draw_geometry()


def get_wrapper(entity: Entity, **kwargs) -> EntityWrapper:
    closest_class = None
    min_distance = 10
    wrappers = {w for w in EntityWrapper.subclasses}

    for wrapper in wrappers:
        distance = 0
        cls = entity.__class__
        while cls != object:
            if cls == wrapper.TYPE:
                if distance < min_distance:
                    min_distance = distance
                    closest_class = wrapper
                break
            cls = cls.__base__
            distance += 1

    if closest_class is None:
        raise TypeError(f'Wrapper not found for {entity}')

    return closest_class(entity, **kwargs)
