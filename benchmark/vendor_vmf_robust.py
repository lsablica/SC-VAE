import math
from numbers import Number

import torch
from torch.distributions.kl import register_kl

DEBUG_VMF_ENTROPY = False    # controls whether we use entropy_debug()


# ============================================================
# Robust special-function backend (movMF-style, torch-only)
# Replaces SciPy ive/ive-ratios with:
#   - Perron continued fraction for I_v(z)/I_{v-1}(z)
#   - log I_v(z) via 0F1 series in log-space
# ============================================================

def bessel_ratio_perron_Iv_Ivminus1(v: Number, z: torch.Tensor,
                                   tol: float = 1e-6,
                                   maxiter: int = 20000) -> torch.Tensor:
    """
    Compute I_v(z) / I_{v-1}(z) via the Perron continued fraction
    (same idea as movMF's mycfP), using only torch ops.

    v: scalar order (Number)
    z: tensor (GPU ok)
    """
    if not isinstance(v, Number):
        raise TypeError("v must be a scalar (Number)")
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor")

    dtype = z.dtype
    device = z.device

    # scalar v as tensor for broadcasting
    v_t = torch.tensor(v, dtype=dtype, device=device)

    # Very light guard for division by zero (not a "robustness trick", just safety)
    eps = torch.finfo(dtype).tiny
    z = torch.clamp(z, min=eps)

    xP = z / 2.0
    p = z / (z + 2.0 * v_t)
    s = p.clone()

    vv = v_t + z + 0.5
    u = (v_t + z) * vv
    w = xP * (v_t + 0.5)

    rho = w / ((v_t + xP) * vv - w)
    p = p * rho
    s = s + p

    k = 2
    while k < maxiter:
        if torch.all(torch.abs(p) <= tol * torch.abs(s)):
            break
        u = u + vv
        vv = vv + 0.5
        w = w + xP
        t = w * (1.0 + rho)
        rho = t / (u - t)
        p = p * rho
        s = s + p
        k += 1

    return s


def log0f1_series(b: torch.Tensor, x: torch.Tensor,
                  tol: float = 1e-6,
                  maxiter: int = 200000) -> torch.Tensor:
    """
    Compute log( 0F1(b; x) ) for x >= 0 via positive-term series:
      0F1(b; x) = sum_{m>=0} x^m / ((b)_m * m!)

    b: tensor broadcastable to x
    x: tensor (>=0)
    """
    dtype = x.dtype
    device = x.device

    b = b.to(dtype=dtype, device=device)
    x = torch.clamp(x, min=0.0)

    term = torch.ones_like(x)
    s = term.clone()

    m = 1.0
    for _ in range(maxiter):
        denom = (b + (m - 1.0)) * m
        new_term = term * (x / denom)
        new_s = s + new_term

        if torch.all(torch.abs(new_term) <= tol * torch.abs(new_s)):
            s = new_s
            break

        term = new_term
        s = new_s
        m += 1.0

    return torch.log(s)


def log0f1_series_parallel(b: torch.Tensor, x: torch.Tensor, K: int = 256) -> torch.Tensor:
    """
    GPU-friendly log(0F1(b; x)) via truncated series and logsumexp:

      0F1(b; x) = sum_{m=0}^{∞} x^m / ( (b)_m * m! )
      log term_m = m*log(x) - log(m!) - log((b)_m)
                = m*log(x) - lgamma(m+1) - (lgamma(b+m) - lgamma(b))

    b, x: broadcastable tensors, x >= 0
    K: number of terms
    """
    # ensure shapes are broadcastable; we sum over the last axis
    x = torch.clamp(x, min=0.0)

    m = torch.arange(K, device=x.device, dtype=x.dtype)  # (K,)

    # Broadcast: (..., 1) with (K,) -> (..., K)
    x_ = x.unsqueeze(-1)
    b_ = b.unsqueeze(-1)
    m_ = m.view(*([1] * x_.ndim), K)  # not strictly needed; kept explicit

    # log(x^m) = m*log(x). For x=0, define log(x)= -inf but only m=0 term survives.
    logx = torch.log(torch.clamp(x_, min=torch.finfo(x.dtype).tiny))
    log_terms = m * logx - torch.lgamma(m + 1.0) - (torch.lgamma(b_ + m) - torch.lgamma(b_))

    # If x==0, only m=0 term should contribute: log_terms[...,0]=0, others -> -inf.
    # The clamp above makes logx finite; fix with an explicit mask:
    if torch.any(x == 0):
        zero_mask = (x_ == 0)
        # set m>0 terms to -inf where x==0
        log_terms = torch.where(zero_mask & (m > 0), torch.tensor(-float("inf"), device=x.device, dtype=x.dtype), log_terms)
        # ensure m=0 term is exactly 0 where x==0
        log_terms = torch.where(zero_mask & (m == 0), torch.tensor(0.0, device=x.device, dtype=x.dtype), log_terms)

    return torch.logsumexp(log_terms, dim=-1)



def log_besselI_via_0f1(nu: Number, kappa: torch.Tensor,
                        tol: float = 1e-6) -> torch.Tensor:
    """
    Compute log I_nu(kappa) using:
      I_nu(k) = (k/2)^nu / Gamma(nu+1) * 0F1(nu+1; k^2/4)
    so:
      log I_nu(k) = nu*log(k/2) - lgamma(nu+1) + log(0F1(nu+1; k^2/4))
    """
    if not isinstance(nu, Number):
        raise TypeError("nu must be a scalar (Number)")
    if not torch.is_tensor(kappa):
        raise TypeError("kappa must be a torch.Tensor")

    dtype = kappa.dtype
    device = kappa.device

    nu_t = torch.tensor(nu, dtype=dtype, device=device)

    eps = torch.finfo(dtype).tiny
    k = torch.clamp(kappa, min=eps)

    x = (k * k) / 4.0
    b = nu_t + 1.0
    b_full = torch.full_like(k, b)
    
    #log0f1 = log0f1_series(b_full, x, tol=tol)
    log0f1 = log0f1_series_parallel(b_full, x, K=128)
    return nu_t * torch.log(k / 2.0) - torch.lgamma(nu_t + 1.0) + log0f1


# ============================================================
# Distributions (same structure as original, SciPy removed)
# ============================================================

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
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    def entropy(self):
        return self.__log_surface_area()

    def __log_surface_area(self):
        lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2], device=self.device))
        return (
            math.log(2.0)
            + ((self._dim + 1) / 2.0) * math.log(math.pi)
            - lgamma
        )


class VonMisesFisher(torch.distributions.Distribution):
    """
    vMF distribution with Ulrich-style rejection sampling, as in the S-VAE repo.

    SciPy Bessel calls replaced by torch-only:
      - ratio I_{m/2}(kappa) / I_{m/2-1}(kappa) via Perron CF
      - log I_{m/2-1}(kappa) via log(0F1) series + lgamma
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
        # Original: ive(m/2,k)/ive(m/2-1,k)  (scaling cancels in ratio)
        ratio = bessel_ratio_perron_Iv_Ivminus1(self.__m / 2.0, self.scale)
        return self.loc * ratio

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
        Clean entropy implementation.

        Same formula as the original repo, but uses robust Bessel ratio/logI.
        """
        ratio = bessel_ratio_perron_Iv_Ivminus1(self.__m / 2.0, self.scale)
        output = -self.scale * ratio
        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def entropy_debug(self):
        """
        Debug version of entropy. Prints diagnostics about NaNs/Infs.
        """
        m = self.__m
        kappa = self.scale

        ratio = bessel_ratio_perron_Iv_Ivminus1(m / 2.0, kappa)
        logI = log_besselI_via_0f1(m / 2.0 - 1.0, kappa)

        print("\n[entropy] m =", m, "  kappa range =",
              float(kappa.min()), "to", float(kappa.max()))
        print("[entropy] ratio min =", float(ratio.min()), "max =", float(ratio.max()))
        print("[entropy] ratio any NaN =", bool(torch.isnan(ratio).any()),
              " any Inf =", bool(torch.isinf(ratio).any()))

        log_norm = self._log_normalization()
        print("[entropy] log_norm any NaN =", bool(torch.isnan(log_norm).any()),
              " any Inf =", bool(torch.isinf(log_norm).any()))

        ent = (-kappa * ratio).view(*(kappa.shape[:-1])) + log_norm
        print("[entropy] entropy any NaN =", bool(torch.isnan(ent).any()),
              " any Inf =", bool(torch.isinf(ent).any()))

        # Optional: show logI diagnostics
        print("[entropy] logI any NaN =", bool(torch.isnan(logI).any()),
              " any Inf =", bool(torch.isinf(logI).any()))

        return ent

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        """
        Original implementation used:
          scale + log( ive(nu, scale) )
        which equals log I_nu(scale), since ive(nu,k)=exp(-k) I_nu(k).

        Here we compute log I_nu(scale) directly and keep the same algebra.
        """
        nu = self.__m / 2.0 - 1.0
        logI = log_besselI_via_0f1(nu, self.scale)

        output = -(
            (self.__m / 2.0 - 1.0) * torch.log(self.scale)
            - (self.__m / 2.0) * math.log(2.0 * math.pi)
            - logI
        )
        return output.view(*(output.shape[:-1]))

    def log_prob(self, value):
        return self._log_unnormalized_prob(value) - self._log_normalization()


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy() + hyu.entropy()


def kl_vmf_official(vmf_dist, hyu_dist):
    return -vmf_dist.entropy() + hyu_dist.entropy()
