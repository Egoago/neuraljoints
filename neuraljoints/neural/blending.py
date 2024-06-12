from neuraljoints.geometry.implicit import Implicit
from neuraljoints.neural.autograd import gradient, hessian
from neuraljoints.neural.embedding import ImplicitEmbedding
from neuraljoints.neural.model import Network
from neuraljoints.neural.trainer import Trainer
from neuraljoints.utils.parameters import BoolParameter


class BlendingNetwork(Network):
    def __init__(self, implicits: list[Implicit], boundaries: list[Implicit], **kwargs):
        self.gradient = BoolParameter('use gradient', False)
        self.implicits = implicits
        self.boundaries = boundaries
        super().__init__(**kwargs)

    def build(self):
        self.embedding = ImplicitEmbedding(self.implicits, self.gradient.value)
        return super().build()


class BlendingTrainer(Trainer):
    def __init__(self, model: BlendingNetwork, **kwargs):
        super().__init__(model=model, implicit=Implicit(name='Dummy'), **kwargs)
        self.implicits = model.implicits
        self.boundaries = model.boundaries

    def step(self):
        outputs = self.sampler()
        x = outputs['x']
        x.requires_grad = self.loss_fn.req_grad

        y_gts = [implicit(x) for implicit in self.implicits]

        if 'grad_gt' in self.loss_fn.attributes:
            grad_gts = [gradient(y_gt, x) for y_gt in y_gts]

        x.requires_grad = self.loss_fn.req_grad or self.sampler.req_grad

        outputs['y_pred'] = self.model(x)

        if 'grad_pred' in self.loss_fn.attributes or 'grad_pred' in self.sampler.attributes:
            outputs['grad_pred'] = gradient(outputs['y_pred'], x)

        if 'hess_pred' in self.loss_fn.attributes:
            outputs['hess_pred'] = hessian(outputs['grad_pred'], x)

        loss = 0.
        for loss_fn in self.loss_fn.losses:
            if bool({'grad_gt', 'y_gt'} & loss_fn.attributes):
                if 'grad_gt' in loss_fn.attributes:
                    for y_gt, grad_gt in zip(y_gts, grad_gts):
                        loss = loss + loss_fn(y_gt=y_gt, grad_gt=grad_gt, **outputs)
                else:
                    for y_gt in y_gts:
                        loss = loss + loss_fn(y_gt=y_gt, **outputs)
            else:
                loss = loss + loss_fn(**outputs)
        outputs['loss'] = loss

        return outputs
