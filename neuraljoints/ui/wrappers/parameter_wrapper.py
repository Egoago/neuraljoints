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
        _, value = imgui.Checkbox('', param.value)
        changed = value != param.value
        param.value = value
        if param.name is not None:
            imgui.SameLine()
            imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.ChoiceParameter)
    def __draw(cls, param: params.ChoiceParameter) -> bool:
        index = param.choices.index(param.value)

        imgui.SetNextItemWidth(200)
        changed, index = imgui.Combo('', index, param.choices)
        if changed:
            param.value = param.choices[index]

        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True

        if param.name is not None:
            imgui.SameLine()
            imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    def __draw_slider(cls, param, value, changed):
        if changed and (not imgui.IsAnyItemActive() or imgui.IsAnyMouseDown()):
            param.value = value

        imgui.SameLine()
        if imgui.Button("reset"):
            param.reset()
            changed = True

        if param.name is not None:
            imgui.SameLine()
            imgui.TextWrapped(f'{param.name:15}')
        return changed

    @classmethod
    @multidispatch(params.FloatParameter)
    def __draw(cls, param: params.FloatParameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat('', param.value, format='%.5f',
                                           v_min=param.min, v_max=param.max)
        return cls.__draw_slider(param, value, changed)

    @classmethod
    @multidispatch(params.IntParameter)
    def __draw(cls, param: params.IntParameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderInt('', param.value, v_min=param.min, v_max=param.max)
        return cls.__draw_slider(param, value, changed)

    @classmethod
    @multidispatch(params.Float3Parameter)
    def __draw(cls, param: params.Float3Parameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderFloat3('', param.value.tolist(),
                                            v_min=param.min, v_max=param.max)
        return cls.__draw_slider(param, value, changed)

    @classmethod
    @multidispatch(params.Int3Parameter)
    def __draw(cls, param: params.Int3Parameter) -> bool:
        imgui.SetNextItemWidth(200)
        changed, value = imgui.SliderInt3('', param.value.tolist(),
                                            v_min=param.min, v_max=param.max)
        return cls.__draw_slider(param, value, changed)

    @classmethod
    @multidispatch(params.Transform)
    def __draw(cls, param: params.Transform) -> bool:
        changed = False
        imgui.SameLine()

        if imgui.Button('transform'):
            imgui.OpenPopup(param.id)
        if imgui.BeginPopup(param.id):
            changed |= cls.draw(param.translation)
            changed |= cls.draw(param.rotation)
            imgui.EndPopup()
        return changed
