import numpy as np
import polyscope as ps

from neuraljoints.geometry.implicit import ImplicitProxy
from neuraljoints.geometry.parametric import Parametric
from neuraljoints.utils.math import normalize
from neuraljoints.utils.parameters import IntParameter, FloatParameter, BoolParameter


class ParametricToImplicitBrute(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.resolution = IntParameter('resolution', value=100, min=3, max=200)

    def forward(self, position):
        parameters = np.linspace(0., 1., self.resolution.value, dtype=np.float32)
        points = self.parametric(parameters)
        return np.linalg.norm(position[:, None, :] - points[None, ...], axis=-1).min(axis=-1)

    def gradient(self, position):
        raise NotImplementedError()


class ParametricToImplicitGWN(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.resolution = IntParameter('resolution', value=10, min=3, max=200)

    @staticmethod
    def angle_between(a, b):
        return

    def forward(self, position):
        parameters = np.linspace(0., 1., self.resolution.value, dtype=np.float32)
        points = self.parametric(parameters)
        d = points[None, ...] - position[:, None, :]
        # udf = np.linalg.norm(d, axis=-1).min(axis=-1)
        d = normalize(d)
        a = d[:, :-1]
        b = d[:, 1:]
        c = np.cross(a, b)
        angles = np.arctan2(np.linalg.norm(c, axis=-1),
                            np.einsum('ijk,ijk->ij', a, b))
        return angles.sum(axis=-1)/(2*np.pi)

    def gradient(self, position):
        raise NotImplementedError()


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
            return np.einsum('ij,ij->i', a, b)

        # brute-force init
        u = np.linspace(-1e-6, 1 + 1e-6, self.resolution.value, dtype=np.float32)
        u[1] = 0
        u[-2] = 1
        c = self.parametric(u)
        d = c[:, None, :] - p[None, ...]
        u = u[np.argmin(np.einsum('ijk,ijk->ij', d, d), axis=0)]

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
        return np.linalg.norm(p - points, axis=-1)

    def gradient(self, position):
        raise NotImplementedError()


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
        return np.einsum('ij,ij->i', a, b)

    def forward(self, p):

        # brute-force init
        t = np.linspace(0, 1, self.resolution.value, dtype=np.float32)
        c = self.parametric(t)
        d = c[:, None, :] - p[None, ...]
        d = np.einsum('ijk,ijk->ij', d, d)
        min_idx = np.argmin(d, axis=0)
        i_a = (min_idx - 1).clip(0, None)
        i_b = (min_idx + 1).clip(None, len(t) - 1)

        t_a, t_b = t[i_a], t[i_b]
        d_a = np.linalg.norm(c[i_a] - p, axis=-1)
        d_b = np.linalg.norm(c[i_b] - p, axis=-1)

        for _ in range(self.iterations.value):
            t_c = (t_a + t_b) / 2
            c = self.parametric(t_c)
            d_c = np.linalg.norm((c - p), axis=-1)

            mask = d_a < d_b
            t_a[~mask] = t_c[~mask]
            d_a[~mask] = d_c[~mask]
            t_b[mask] = t_c[mask]
            d_b[mask] = d_c[mask]

        return np.where(d_a < d_b, d_a, d_b)

    def gradient(self, position):
        raise NotImplementedError()


class ParametricToImplicitNewtonBinary(ImplicitProxy):
    def __init__(self, parametric: Parametric, **kwargs):
        super().__init__(**kwargs)
        self.parametric = parametric
        self.iterations = IntParameter('iterations', value=3, min=0, max=300)
        self.resolution = IntParameter('resolution', value=20, min=2, max=40)
        self.band = FloatParameter('band', value=0.1, min=0.001, max=1)

    def forward(self, p):
        def dot(a, b):
            return np.einsum('ij,ij->i', a, b)

        # brute-force init
        u = np.linspace(0, 1, self.resolution.value, dtype=np.float32)
        c = self.parametric(u)
        d = c[:, None, :] - p[None, ...]
        u = u[np.argmin(np.einsum('ijk,ijk->ij', d, d), axis=0)]
        u_a = u - self.band.value / 2
        u_b = u + self.band.value / 2

        u_ab = np.concatenate([u_a, u_b])
        c_a, c_b = np.split(self.parametric(u_ab), 2)
        cd_a, cd_b = np.split(self.parametric.gradient(u_ab), 2)
        f_a = dot(cd_a, c_a - p)
        f_b = dot(cd_b, c_b - p)

        # switch
        mask = f_a < f_b
        u_c = np.where(mask, u_a, u_b)
        u_b = np.where(~mask, u_a, u_b)
        u_a = u_c
        f_c = np.where(mask, f_a, f_b)
        f_b = np.where(~mask, f_a, f_b)
        f_a = f_c

        mask = f_a < 0
        for _ in range(self.iterations.value):
            u = (u_a[mask] + u_b[mask]) / 2
            c = self.parametric(u)
            cd = self.parametric.gradient(u)
            f = dot(cd, c - p[mask])

            negative_f = (f_a[mask] < f) * (f < 0)
            u_a[mask] = np.where(negative_f, u, u_a[mask])
            f_a[mask] = np.where(negative_f, f, f_a[mask])
            c_a[mask] = np.where(negative_f[:, None], c, c_a[mask])

            positive_f = (0 < f) * (f < f_b[mask])
            u_b[mask] = np.where(positive_f, u, u_b[mask])
            f_b[mask] = np.where(positive_f, f, f_b[mask])
            c_b[mask] = np.where(positive_f[:, None], c, c_b[mask])

            mask[mask] = positive_f + negative_f

        c = np.where((np.abs(f_a) < np.abs(f_b))[:, None], c_a, c_b)
        return np.linalg.norm(c - p, axis=-1)

    def gradient(self, position):
        raise NotImplementedError()
