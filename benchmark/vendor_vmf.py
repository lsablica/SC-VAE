import math
from numbers import Number

import numpy as np
import torch
from torch.distributions.kl import register_kl
import scipy.special

DEBUG_VMF_IVE = False        # controls prints in IveFunction.forward
DEBUG_VMF_ENTROPY = False    # controls whether we use entropy_debug()


class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, z):
        """
        Wraps scipy.special.ive on CPU, just like the original implementation.
        v: scalar (Number)
        z: tensor (can be on GPU)
        """
        assert isinstance(v, Number), "v must be a scalar"
        ctx.save_for_backward(z)
        ctx.v = v

        z_cpu = z.detach().cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu)
        else:
            output = scipy.special.ive(v, z_cpu)

        # Only do expensive printing if debug is enabled
        if DEBUG_VMF_IVE:
            out_min = output.min()
            out_max = output.max()
            out_abs_min = np.abs(output).min()
            if (out_abs_min == 0.0) or (not np.isfinite(output).all()):
                print(f"[IveFunction] v={v}, z range=({z_cpu.min()}, {z_cpu.max()})")
                print(
                    f"[IveFunction] output range=({out_min}, {out_max}), "
                    f"min |output|={out_abs_min}, any non-finite={~np.isfinite(output).all()}"
                )

        return torch.tensor(output, device=z.device, dtype=z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1]
        v = ctx.v
        # d/dz ive(v, z) = ive(v-1, z) - ive(v, z) * (v + z) / z
        return (
            None,
            grad_output * (ive(v - 1, z) - ive(v, z) * (v + z) / z),
        )


def ive(v, z):
    """
    Exponentially scaled modified Bessel of the first kind, order v.
    v: scalar
    z: tensor
    """
    return IveFunction.apply(v, z)



class HypersphericalUniform(torch.distributions.Distribution):
    """
    Uniform on S^d, following their entropy implementation.
    Note: In the original code, dim argument is (z_dim - 1).
    """

    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    def __init__(self, dim, validate_args=None, device="cpu"):
        super().__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(device)
        )

    def entropy(self):
        return self.__log_surface_area()

    def __log_surface_area(self):
        # surface area of S^dim
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(
                torch.tensor([(self._dim + 1) / 2], device=self.device)
            )
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2]).to(self.device)
            )

        return (
            math.log(2.0)
            + ((self._dim + 1) / 2.0) * math.log(math.pi)
            - lgamma
        )



class VonMisesFisher(torch.distributions.Distribution):
    """
    vMF distribution with Ulrich-style rejection sampling, as in the S-VAE repo.
    """

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # Uses IveFunction for the Bessel ratio, same as original
        return self.loc * (
            ive(self.__m / 2.0, self.scale)
            / ive(self.__m / 2.0 - 1.0, self.scale)
        )

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=20):
        """
        loc: (..., m) unit vectors
        scale: (..., 1) concentration κ
        k: number of proposals per while-loop iteration
        """
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = torch.tensor(
            [1.0] + [0.0] * (self.__m - 1),
            device=self.device,
            dtype=self.dtype,
        )
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        """
        Reparameterized sampling via rejection sampling, using Ulrich (1984) style.
        """
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        # For m == 3 they use a special case; here we keep full version for all m>=3
        w = self.__sample_w_rej(shape=shape)

        # Sample v on S^{m-2}
        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1.0 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), dim=-1)
        z = self.__householder_rotation(x)
        return z.type(self.dtype)

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # Taylor approximation blending for large κ, like original
        b_app = (self.__m - 1) / (4.0 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10.0,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1.0 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4.0
        d = (
            4.0 * a * b / (1.0 + b)
            - (self.__m - 1)
            * math.log(float(self.__m - 1))
        )

        self.__b, (self.__e, self.__w) = b, self.__while_loop(
            b, a, d, shape, k=self.k
        )
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=dim),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        """
        Matrix while loop: samples a matrix of [A, k] candidates, to avoid Python looping
        over individual components – but still a Python loop over rejection iterations.
        """
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w = torch.zeros_like(b, device=self.device, dtype=self.dtype)
        e = torch.zeros_like(b, device=self.device, dtype=self.dtype)
        bool_mask = (torch.ones_like(b, dtype=torch.bool)).to(self.device)

        sample_shape = torch.Size([b.shape[0], k])
        full_shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con = torch.tensor(
                (self.__m - 1) / 2.0,
                dtype=torch.float64,
                device=self.device,
            )
            e_ = (
                torch.distributions.Beta(con, con)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0.0 + eps, 1.0 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1.0 - (1.0 + b) * e_) / (1.0 - (1.0 - b) * e_)
            t = (2.0 * a * b) / (1.0 - (1.0 - b) * e_)

            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)

            w_g = w_.gather(1, accept_idx_clamped)
            e_g = e_.gather(1, accept_idx_clamped)

            reject = accept_idx < 0
            if torch.__version__ >= "1.2.0":
                accept_mask = ~reject
            else:
                accept_mask = 1 - reject

            # update accepted positions
            active = bool_mask * accept_mask
            w[active] = w_g[active]
            e[active] = e_g[active]

            # mark those as done
            bool_mask[active] = False

        return e.reshape(full_shape), w.reshape(full_shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2.0 * (x * u).sum(-1, keepdim=True) * u
        return z
        
    def entropy(self):
        """
        Clean entropy implementation
        Same formula as the original repo, but without any printing.
        """
        output = -self.scale * ive(self.__m / 2.0, self.scale) / ive(
            (self.__m / 2.0) - 1.0, self.scale
        )
        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def entropy_debug(self):
        """
        Debug version of entropy. Prints detailed diagnostics about Ive underflow,
        NaNs, and Infs. 
        """
        m = self.__m
        kappa = self.scale

        ive_m2 = ive(m / 2.0, kappa)
        ive_m2m1 = ive(m / 2.0 - 1.0, kappa)

        print("\n[entropy] m =", m, "  kappa range =",
              float(kappa.min()), "to", float(kappa.max()))
        print("[entropy] ive(m/2)    min =", float(ive_m2.min()),
              "max =", float(ive_m2.max()))
        print("[entropy] ive(m/2-1)  min =", float(ive_m2m1.min()),
              "max =", float(ive_m2m1.max()))
        print("[entropy] ive(m/2-1) any zero =", bool((ive_m2m1 == 0).any()))
        print("[entropy] ive(m/2-1) any NaN =", bool(torch.isnan(ive_m2m1).any()))
        print("[entropy] ive(m/2-1) any Inf =", bool(torch.isinf(ive_m2m1).any()))

        ratio = ive_m2 / ive_m2m1
        print("[entropy] ratio any NaN =", bool(torch.isnan(ratio).any()),
              " any Inf =", bool(torch.isinf(ratio).any()))

        output = -kappa * ratio

        log_norm = self._log_normalization()
        print("[entropy] log_norm any NaN =", bool(torch.isnan(log_norm).any()),
              " any Inf =", bool(torch.isinf(log_norm).any()))

        ent = output.view(*(output.shape[:-1])) + log_norm

        print("[entropy] entropy any NaN =", bool(torch.isnan(ent).any()),
              " any Inf =", bool(torch.isinf(ent).any()))

        return ent


    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = -(
            (self.__m / 2.0 - 1.0) * torch.log(self.scale)
            - (self.__m / 2.0) * math.log(2.0 * math.pi)
            - (
                self.scale
                + torch.log(ive(self.__m / 2.0 - 1.0, self.scale))
            )
        )
        return output.view(*(output.shape[:-1]))

    def log_prob(self, value):
        return self._log_unnormalized_prob(value) - self._log_normalization()


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy() + hyu.entropy()


def kl_vmf_official(vmf_dist, hyu_dist):
    return -vmf_dist.entropy() + hyu_dist.entropy()

