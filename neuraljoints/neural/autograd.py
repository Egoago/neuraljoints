import torch


def gradient(outputs, inputs, create_graph=True, retain_graph=True):
    ones = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    return torch.autograd.grad(outputs, inputs, grad_outputs=ones, create_graph=create_graph, retain_graph=retain_graph)[0]


def hessian(gradients, inputs):
    grad_grads = []
    for i in range(gradients.shape[-1]):
        grad_grads.append(gradient(gradients[..., i], inputs))
    return torch.stack(grad_grads, dim=-2)
