from neuraljoints.utils.parameters import FloatParameter


class LRScheduler:
    def __init__(self, optimizer, parameter: FloatParameter):
        self.optimizer = optimizer
        self.parameter = parameter

    def update(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    @property
    def lr(self):
        return self.parameter.value
