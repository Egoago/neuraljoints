
class Drawable:
    def __init__(self, **kwargs):
        super().__init__()
        self.changed = False
        self.startup = True

    def draw(self, refresh=False):
        self.changed = False

        self.draw_ui()

        if self.changed or self.startup or refresh:
            self.draw_geometry()

    def draw_geometry(self):
        pass

    def draw_ui(self):
        self.changed = self.startup
        self.startup = False
