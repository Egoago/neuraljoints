import unittest

import torch

from neuraljoints.neural.autograd import gradient, hessian


class TestAutograd(unittest.TestCase):
    EPSILON = 1e-7
    N = 1000
    D = 3
    dtype = torch.float64

    def func(self, x):
        return (torch.sin(x**2)-1).sum(-1)

    @torch.no_grad()
    def get_offsets(self, x, epsilon):
        offset = torch.ones_like(x[..., 0]) * epsilon
        x_offsets = []
        for dim in range(x.shape[-1]):
            x_offset = x.clone()
            x_offset[..., dim] = x_offset[..., dim] + offset
            x_offsets.append(x_offset)
        for dim in range(x.shape[-1]):
            x_offset = x.clone()
            x_offset[..., dim] = x_offset[..., dim] - offset
            x_offsets.append(x_offset)
        return torch.stack(x_offsets, dim=0)

    @torch.no_grad()
    def grad_num(self, x):
        x_offsets = self.get_offsets(x, self.EPSILON)
        y_offsets = self.func(x_offsets)
        y_upper = y_offsets[:x.shape[-1]]
        y_lower = y_offsets[x.shape[-1]:]
        return (y_upper - y_lower).T / (self.EPSILON * 2.)

    @torch.no_grad()
    def hess_num(self, x):
        epsilon = self.EPSILON * 10000
        x_offsets = self.get_offsets(x, epsilon)
        grad_nums = []
        for x_offset in x_offsets:
            grad_nums.append(self.grad_num(x_offset))
        grad_nums = torch.stack(grad_nums, dim=1)
        grad_nums_upper = grad_nums[:, :x.shape[-1]]
        grad_nums_lower = grad_nums[:, x.shape[-1]:]
        return (grad_nums_upper - grad_nums_lower) / (epsilon * 2.)

    def test_gradient(self):
        for device in ['cpu', 'cuda']:
            with self.subTest(f'device={device}'):
                x = torch.rand((self.N, self.D), dtype=self.dtype, device=device, requires_grad=True)
                y = self.func(x)

                grad_auto = gradient(y, x).detach().cpu()

                grad_num = self.grad_num(x.detach().cpu())

                equals = torch.isclose(grad_num, grad_auto)
                self.assertEqual(len(equals), equals.prod(dim=-1).sum())

    def test_hessian(self):
        for device in ['cpu', 'cuda']:
            with self.subTest(f'device={device}'):
                x = torch.rand((self.N, self.D), dtype=self.dtype, device=device, requires_grad=True)
                y = self.func(x)

                grad_auto = gradient(y, x)
                hess_auto = hessian(grad_auto, x).detach().cpu()

                hess_num = self.hess_num(x.detach().cpu())

                abs_error = (hess_auto - hess_num).abs().sum(-1).sum(-1)
                equals = abs_error < 1e-4
                self.assertEqual(len(equals), equals.sum())


if __name__ == '__main__':
    unittest.main()
