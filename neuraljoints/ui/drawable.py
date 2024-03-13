from abc import abstractmethod, ABC


class Drawable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def draw(self) -> bool:
        pass
