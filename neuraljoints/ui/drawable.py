
class Drawable:
    def __init__(self):
        super().__init__()
        self.changed = False
        self.startup = True

    def draw(self, refresh=False):
        self.__set_changed(False)
        self.draw_ui()
        for value in self.__dict__.values():
            if isinstance(value, Drawable):
                value.draw()
                self.changed = value.changed or self.changed
        if self.changed or refresh:
            self.draw_geometry()

    def draw_geometry(self):
        pass

    def draw_ui(self):
        self.changed = self.startup
        self.startup = False

    def __set_changed(self, changed):
        self.changed = changed
        for value in self.__dict__.values():
            if isinstance(value, Drawable):
                value.__set_changed(changed)
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, Drawable):
                        v.__set_changed(changed)
