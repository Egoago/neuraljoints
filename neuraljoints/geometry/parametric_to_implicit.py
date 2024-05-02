import torch

from neuraljoints.geometry.implicit import ImplicitProxy
from neuraljoints.geometry.parametric import Parametric
from neuraljoints.utils.parameters import IntParameter


class ParametricToImplicitBrute(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.resolution = IntParameter('resolution', value=100, min=3, max=200)

    def forward(self, position):
        parameters = torch.linspace(0., 1., self.resolution.value,
                                    dtype=position.dtype, device=position.device)
        points = self.parametric(parameters)
        return torch.linalg.norm(position[:, None, :] - points[None, ...], dim=-1).amin(dim=-1)


class ParametricToImplicitGWN(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.resolution = IntParameter('resolution', value=10, min=3, max=200)

    def forward(self, position):
        parameters = torch.linspace(0., 1., self.resolution.value,
                                    dtype=position.dtype, device=position.device)
        points = self.parametric(parameters)
        d = points[None, ...] - position[:, None, :]
        # udf = torch.linalg.norm(d, dim=-1).amin(dim=-1)
        d = torch.nn.functional.normalize(d, dim=-1)
        a = d[:, :-1]
        b = d[:, 1:]
        c = torch.cross(a, b)
        angles = torch.arctan2(torch.linalg.norm(c, dim=-1),
                               torch.einsum('ijk,ijk->ij', a, b))
        return angles.sum(dim=-1)/(2*torch.pi)


class ParametricToImplicitNewton(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.iterations = IntParameter('iterations', value=3, min=0, max=300)
        self.resolution = IntParameter('resolution', value=100, min=2, max=40)
        # self.distance_tol = FloatParameter('distance tolerance', value=0, min=0, max=0.1)
        # self.cosine_tol = FloatParameter('cosine tolerance', value=0, min=0, max=1)

    def forward(self, p):
        def dot(a, b):
            return torch.einsum('ij,ij->i', a, b)

        # brute-force init
        u = torch.linspace(-1e-6, 1 + 1e-6, self.resolution.value, dtype=p.dtype, device=p.device)
        u[1] = 0
        u[-2] = 1
        c = self.parametric(u)
        d = c[:, None, :] - p[None, ...]
        u = u[torch.argmin(torch.einsum('ijk,ijk->ij', d, d), dim=0)]

        mask = (u <= 1) * (u >= 0)
        for _ in range(self.iterations.value):
            u_masked = u[mask]
            c = self.parametric(u_masked)
            c_d = self.parametric.gradient(u_masked)
            c_d_d = self.parametric.gradgrad(u_masked)
            d = c - p[mask]

            u[mask] = u_masked - (dot(c_d, d) / (dot(c_d_d, d) + dot(c_d, c_d)))
            # u = u.clip(0, 1)

            # d_norm = np.linalg.norm(d, axis=-1)
            # c_d_norm = np.linalg.norm(c_d, axis=-1)
            # mask[mask] = d_norm > self.distance_tol.value
            # mask = mask * (dot(c_d, d) / (c_d_norm * d_norm) > self.cosine_tol.value)

            # if not np.any(mask):
            #     break
        points = self.parametric(u)
        return torch.linalg.norm(p - points, dim=-1)


class ParametricToImplicitBinary(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.iterations = IntParameter('iterations', value=5, min=0, max=100)
        self.resolution = IntParameter('resolution', value=50, min=2, max=20)

    @staticmethod
    def dot(a, b=None):
        if b is None:
            b = a
        return torch.einsum('ij,ij->i', a, b)

    def forward(self, p):

        # brute-force init
        t = torch.linspace(0, 1, self.resolution.value, dtype=p.dtype, device=p.device)
        c = self.parametric(t)
        d = c[:, None, :] - p[None, ...]
        d = torch.einsum('ijk,ijk->ij', d, d)
        min_idx = torch.argmin(d, dim=0)
        i_a = (min_idx - 1).clip(0, None)
        i_b = (min_idx + 1).clip(None, len(t) - 1)

        t_a, t_b = t[i_a], t[i_b]
        d_a = torch.linalg.norm(c[i_a] - p, dim=-1)
        d_b = torch.linalg.norm(c[i_b] - p, dim=-1)

        for _ in range(self.iterations.value):
            t_c = (t_a + t_b) / 2
            c = self.parametric(t_c)
            d_c = torch.linalg.norm((c - p), dim=-1)

            mask = d_a < d_b
            t_a[~mask] = t_c[~mask]
            d_a[~mask] = d_c[~mask]
            t_b[mask] = t_c[mask]
            d_b[mask] = d_c[mask]

        return torch.where(d_a < d_b, d_a, d_b)
