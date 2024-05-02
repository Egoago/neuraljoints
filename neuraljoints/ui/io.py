from polyscope import imgui


class IOListener:
    io_listeners = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        IOListener.io_listeners.append(self)

    def on_mouse_clicked(self, screen_coords, button):
        pass

    def on_mouse_double_clicked(self, screen_coords, button):
        pass

    def on_mouse_released(self, screen_coords, button):
        pass

    def on_mouse_down(self, screen_coords, button):
        pass


class IOHandler:
    @classmethod
    def update(cls, io):
        screen_coords = io.MousePos
        for button in range(imgui.ImGuiMouseButton_COUNT):
            if imgui.IsMouseClicked(button):
                [l.on_mouse_clicked(screen_coords, button) for l in IOListener.io_listeners]
            if imgui.IsMouseDoubleClicked(button):
                [l.on_mouse_double_clicked(screen_coords, button) for l in IOListener.io_listeners]
            if imgui.IsMouseReleased(button):
                [l.on_mouse_released(screen_coords, button) for l in IOListener.io_listeners]
            if imgui.IsMouseDown(button):
                [l.on_mouse_down(screen_coords, button) for l in IOListener.io_listeners]
