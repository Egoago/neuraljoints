import sys
import time
from abc import ABCMeta, ABC
from inspect import isabstract
from io import StringIO


class RegisteredMeta(ABCMeta):
    def __new__(cls, name, bases, dct):

        def get_subclass(cls, name):
            for c in cls.subclasses:
                if c.__name__ == name:
                    return c
            return None

        cls = super().__new__(cls, name, bases, dct)
        if not hasattr(cls, 'registry'):
            cls.registry = set()
            cls.subclasses = classmethod(property(lambda cls: {c for c in cls.registry if issubclass(c, cls)}))
            cls.subclass_names = classmethod(property(lambda cls: {c.__name__ for c in cls.subclasses}))
            cls.get_subclass = classmethod(get_subclass)
        if not isabstract(cls) and cls != ABC:
            cls.registry.add(cls)
        return cls


def redirect_stdout() -> StringIO:
    class DualOutput:
        def __init__(self):
            self.stdout = sys.stdout
            self.stringio = StringIO()

        def write(self, text):
            self.stdout.write(text)
            self.stringio.write(text)

        def flush(self):
            self.stdout.flush()
            self.stringio.flush()

    dual_output = DualOutput()
    sys.stdout = dual_output
    return dual_output.stringio


class FPSCounter:
    BUFFER_SIZE = 3

    def __init__(self):
        self.prev_time = time.time_ns()
        self.prev_fps = []

    def update(self) -> (float, float):
        new_time = time.time_ns()
        fps = 1e9 / (new_time - self.prev_time)
        self.prev_fps.append(fps)
        self.prev_fps = self.prev_fps[FPSCounter.BUFFER_SIZE:]
        return fps, sum(self.prev_fps) / len(self.prev_fps)
