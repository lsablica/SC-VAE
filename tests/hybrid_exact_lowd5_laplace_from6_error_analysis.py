
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import hyp2f1, digamma
from scipy.optimize import minimize_scalar

def w_d(d: int) -> float:
    delta = (d - 1) / 2.0
    return float(digamma(d - 1) - digamma(delta) - np.log(2.0))

def H_exact_low_d(z, d: int):
    z = np.asarray(z, dtype=float)
    a = np.sqrt(1.0 - z)

    if d == 2:
        return 2.0 * np.log((1.0 + a) / 2.0)
    elif d == 3:
        return -1.0 - (1.0 - z) * np.log(1.0 - z) / z
    elif d == 4:
        return 2.0 * np.log((1.0 + a) / 2.0) + ((1.0 - a) ** 2) / (2.0 * (1.0 + a) ** 2)
    elif d == 5:
        return 2.0 / z**2 - 2.0 / z - 5.0 / 6.0 + ((z + 2.0) * (1.0 - z) ** 2 / z**3) * np.log(1.0 - z)
    else:
        raise ValueError("Exact low-dimensional formula implemented only for d = 2, 3, 4, 5.")

def H_true(z, d: int, h: float = 1e-5):
    z = np.asarray(z, dtype=float)
    if d in (2, 3, 4, 5):
        return H_exact_low_d(z, d)

    delta = (d - 1) / 2.0
    c = d - 1.0
    lp = np.log(hyp2f1(c + h, delta, c, z))
    lm = np.log(hyp2f1(c - h, delta, c, z))
    D = (lp - lm) / (2.0 * h)
    return D + np.log(1.0 - z)

def H_laplace(z, d: int):
    z = np.asarray(z, dtype=float)
    return np.log(1.0 - z / 2.0) - w_d(d) * (z / (2.0 - z)) ** 2

def H_hybrid(z, d: int):
    if d in (2, 3, 4, 5):
        return H_exact_low_d(z, d)
    return H_laplace(z, d)

def kl_error(z, d: int):
    return (d - 1.0) * (H_true(z, d) - H_hybrid(z, d))

def maximize_abs_kl_error(d: int):
    if d in (2, 3, 4, 5):
        return {
            "d": d,
            "z_star": 0.0,
            "signed_kl_error": 0.0,
            "max_abs_kl_error": 0.0,
            "w_d": w_d(d),
        }

    z_grid = np.r_[np.linspace(1e-6, 0.10, 300),
                   np.linspace(0.10, 0.97, 1200),
                   np.linspace(0.97, 0.995, 200)]
    vals = np.abs(kl_error(z_grid, d))
    i = int(np.nanargmax(vals))
    lo = z_grid[max(i - 2, 0)]
    hi = z_grid[min(i + 2, len(z_grid) - 1)]

    obj = lambda z: -abs(float(kl_error(z, d)))
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11, "maxiter": 400})

    z_star = float(res.x)
    signed_err = float(kl_error(z_star, d))

    return {
        "d": d,
        "z_star": z_star,
        "signed_kl_error": signed_err,
        "max_abs_kl_error": abs(signed_err),
        "w_d": w_d(d),
    }

dims = list(range(2, 201))
df = pd.DataFrame([maximize_abs_kl_error(d) for d in dims])
df.to_csv("hybrid_exact_lowd5_laplace_from6_max_error_vs_d.csv", index=False)

# points for d<6
df_pts = df[df["d"] < 6]
df_line = df[df["d"] >= 6]

plt.figure(figsize=(8, 5))
plt.scatter(df_pts["d"], df_pts["max_abs_kl_error"], label="exact (d=2,...,5)")
plt.plot(df_line["d"], df_line["max_abs_kl_error"], label="Laplace from d=6")
plt.xlabel("d")
plt.ylabel("max |KL error|")
plt.title("Hybrid rule: exact for d=2,3,4,5; Laplace for d≥6")
plt.legend()
plt.tight_layout()
plt.savefig("hybrid_exact_lowd5_laplace_from6_max_kl_error_vs_d.png", dpi=200)

plt.figure(figsize=(8, 5))
plt.scatter(df_pts["d"], df_pts["z_star"], label="exact (d=2,...,5)")
plt.plot(df_line["d"], df_line["z_star"], label="Laplace from d=6")
plt.xlabel("d")
plt.ylabel("z* (argmax of |KL error|)")
plt.title("Location of worst-case z* for the hybrid rule")
plt.legend()
plt.tight_layout()
plt.savefig("hybrid_exact_lowd5_laplace_from6_argmax_vs_d.png", dpi=200)

print(df[df["d"].isin([2,3,4,5,6,7,10,20,50,100,200])].to_string(index=False))
