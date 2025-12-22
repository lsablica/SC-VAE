import importlib
import torch


def row_normalize(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def max_abs_rel(a, b, eps=1e-12):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    abs_err = (a - b).abs().max().item()
    rel_err = ((a - b).abs() / (b.abs() + eps)).max().item()
    return abs_err, rel_err


def report(name, a, b):
    abs_err, rel_err = max_abs_rel(a, b)
    print(f"  {name:18s}  max|Δ|={abs_err:.3e}   max rel={rel_err:.3e}")


def run_dim_test(dim, batch=64, dtype=torch.float32, seed=0):
    print(f"\n=== dim={dim}  batch={batch}  dtype={dtype} ===")

    torch.manual_seed(seed)

    vendor = importlib.import_module("benchmark.vendor_vmf")
    robust = importlib.import_module("benchmark.vendor_vmf_robust")

    device = torch.device("cpu")

    loc = row_normalize(torch.randn(batch, dim, device=device, dtype=dtype))
    scale = torch.full((batch, 1), 10.0, device=device, dtype=dtype)

    vmf_v = vendor.VonMisesFisher(loc, scale)
    vmf_r = robust.VonMisesFisher(loc, scale)

    report("mean", vmf_v.mean, vmf_r.mean)

    ln_v = vmf_v._log_normalization()
    ln_r = vmf_r._log_normalization()
    report("_log_normalization", ln_v, ln_r)

    ent_v = vmf_v.entropy()
    ent_r = vmf_r.entropy()
    report("entropy", ent_v, ent_r)

    x = row_normalize(torch.randn(batch, dim, device=device, dtype=dtype))
    lp_v = vmf_v.log_prob(x)
    lp_r = vmf_r.log_prob(x)
    report("log_prob(x)", lp_v, lp_r)

    for tname, t in [("mean_r", vmf_r.mean), ("ln_r", ln_r), ("ent_r", ent_r), ("lp_r", lp_r)]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"  [WARN] robust produced NaN/Inf in {tname}")

    for tname, t in [("mean_v", vmf_v.mean), ("ln_v", ln_v), ("ent_v", ent_v), ("lp_v", lp_v)]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"  [WARN] vendor (SciPy) produced NaN/Inf in {tname}")


def run_gpu_robust_smoke(dim, batch=64, dtype=torch.float32, seed=0):
    if not torch.cuda.is_available():
        return
    print(f"\n--- GPU smoke (robust only) dim={dim} ---")
    torch.manual_seed(seed)
    robust = importlib.import_module("benchmark.vendor_vmf_robust")
    device = torch.device("cuda")

    loc = row_normalize(torch.randn(batch, dim, device=device, dtype=dtype))
    scale = torch.full((batch, 1), 10.0, device=device, dtype=dtype)
    vmf_r = robust.VonMisesFisher(loc, scale)

    mean_r = vmf_r.mean
    ent_r = vmf_r.entropy()
    x = row_normalize(torch.randn(batch, dim, device=device, dtype=dtype))
    lp_r = vmf_r.log_prob(x)

    ok = (torch.isfinite(mean_r).all() and torch.isfinite(ent_r).all() and torch.isfinite(lp_r).all())
    print("  finite(mean, entropy, log_prob):", bool(ok))


if __name__ == "__main__":
    for d in [8, 16, 32, 64]:
        run_dim_test(d, batch=64, dtype=torch.float32, seed=0)

    for d in [128, 256]:
        try:
            run_dim_test(d, batch=64, dtype=torch.float32, seed=0)
        except Exception as e:
            print(f"\n=== dim={d} vendor comparison failed (sometimes expected) ===")
            print("  error:", repr(e))

        run_gpu_robust_smoke(d, batch=64, dtype=torch.float32, seed=0)
