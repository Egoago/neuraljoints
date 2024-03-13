
class Drawable:
    def __init__(self):
        super().__init__()
        self.changed = False
        self.startup = True

    def draw(self):
        self.draw_ui()
        for value in self.__dict__.values():
            if isinstance(value, Drawable):
                value.draw()
                self.changed = value.changed or self.changed
        if self.changed:
            self.draw_geometry()

    def draw_geometry(self):
        pass

    def draw_ui(self):
        self.changed = self.startup
        self.startup = False
