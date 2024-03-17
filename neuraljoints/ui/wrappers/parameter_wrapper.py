from polyscope import imgui
import neuraljoints.utils.parameters as params
from neuraljoints.utils import multidispatch


class ParameterWrapper:
    @classmethod
    def draw(cls, param: params.Parameter) -> bool:
        imgui.PushId(param.id)
        changed = cls.__draw(param)
        imgui.PopID()
        return changed

    @classmethod
    @multidispatch(params.FloatParameter)
    def __draw(cls, param: params.FloatParameter) -> bool:
        imgui.TextWrapped(f'{param.name:15}')
        imgui.SameLine()
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat('', param.value,
                                           v_min=param.min, v_max=param.max)
        if changed:
            param.value = value
        imgui.SameLine()
        if imgui.Button("reset", (55, 28)):
            param.reset()
            changed = True
        return changed

    @classmethod
    @multidispatch(params.IntParameter)
    def __draw(cls, param: params.IntParameter) -> bool:
        imgui.TextWrapped(f'{param.name:15}')
        imgui.SameLine()
        imgui.SetNextItemWidth(200)
        changed, value = imgui.InputInt('', param.value, step=1)
        # changed, value = imgui.SliderInt('', param.value,
        #                                  v_min=param.min, v_max=param.max)
        if changed:
            param.value = value
        imgui.SameLine()
        if imgui.Button("reset", (55, 28)):
            param.reset()
            changed = True
        return changed

    @classmethod
    @multidispatch(params.Float3Parameter)
    def __draw(cls, param: params.Float3Parameter) -> bool:
        imgui.TextWrapped(f'{param.name:15}')
        imgui.SameLine()
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat3('', param.value.tolist(),
                                            v_min=param.min, v_max=param.max)
        if changed:
            param.value = value
        imgui.SameLine()
        if imgui.Button("reset", (55, 28)):
            param.reset()
            changed = True
        return changed

    @classmethod
    @multidispatch(params.Transform)
    def __draw(cls, param: params.Transform) -> bool:
        changed = False
        if imgui.TreeNode(param.name):
            changed = cls.draw(param.translation)
            changed = cls.draw(param.scale) or changed
            imgui.TreePop()
        return changed
