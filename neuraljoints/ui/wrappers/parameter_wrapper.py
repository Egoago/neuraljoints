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
    @multidispatch(params.BoolParameter)
    def __draw(cls, param: params.BoolParameter) -> bool:
        changed, value = imgui.Checkbox('', param.value)
        if changed:
            param.value = value
        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True
        imgui.SameLine()
        imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.ChoiceParameter)
    def __draw(cls, param: params.ChoiceParameter) -> bool:
        index = param.choices.index(param.value)
        changed, index = imgui.Combo('', index, param.choices)
        if changed:
            param.value = param.choices[index]

        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True

        imgui.SameLine()
        imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.FloatParameter)
    def __draw(cls, param: params.FloatParameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat('', param.value, format='%.5f',
                                           v_min=param.min, v_max=param.max, power=-2)  #TODO
        if changed:
            param.value = value

        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True

        imgui.SameLine()
        imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.IntParameter)
    def __draw(cls, param: params.IntParameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.InputInt('', param.value, step=1)
        # changed, value = imgui.SliderInt('', param.value,
        #                                  v_min=param.min, v_max=param.max)
        imgui.SameLine()
        imgui.TextWrapped(f'{param.name:15}')
        if changed:
            param.value = value
        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True
        return changed

    @classmethod
    @multidispatch(params.Float3Parameter)
    def __draw(cls, param: params.Float3Parameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat3('', param.value.tolist(),
                                            v_min=param.min, v_max=param.max, power=param.power)
        if changed:
            param.value = value

        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True

        imgui.SameLine()
        imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.Transform)
    def __draw(cls, param: params.Transform) -> bool:
        changed = False
        if imgui.TreeNode(param.name):
            changed = cls.draw(param.translation)
            changed = cls.draw(param.rotation) or changed
            changed = cls.draw(param.scale_param) or changed
            imgui.TreePop()
        return changed
