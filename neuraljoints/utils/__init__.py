import functools


def multidispatch(*types):
    # https://stackoverflow.com/questions/25343981/method-overloading-for-different-argument-type-in-python
    def register(function):
        name = function.__name__
        mm = multidispatch.registry.get(name)
        if mm is None:
            @functools.wraps(function)
            def wrapper(self, *args):
                types = tuple(arg.__class__ for arg in args)
                function = wrapper.typemap.get(types)
                if function is None:
                    raise TypeError("no match")
                return function(self, *args)

            wrapper.typemap = {}
            mm = multidispatch.registry[name] = wrapper
        if types in mm.typemap:
            raise TypeError("duplicate registration")
        mm.typemap[types] = function
        return mm

    return register


multidispatch.registry = {}