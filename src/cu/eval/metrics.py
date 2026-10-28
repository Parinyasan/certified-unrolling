from __future__ import annotations

import math

import numpy as np


def nmse_db(x_hat: np.ndarray, x_star: np.ndarray) -> float:
    x_hat = np.asarray(x_hat, np.float64)
    x_star = np.asarray(x_star, np.float64)
    num = np.mean(np.sum((x_hat - x_star) ** 2, axis=-1))
    den = np.mean(np.sum(x_star**2, axis=-1))
    return float(10.0 * np.log10(num / den))


def theorem2_bound(rho: float, d, eps_tol):
    d = np.asarray(d, np.float64)
    eps_tol = np.asarray(eps_tol, np.float64)
    with np.errstate(divide="ignore"):
        val = np.log((1.0 + rho) * d / ((1.0 - rho) * eps_tol)) / math.log(1.0 / rho)
    b = np.ceil(val)
    b = np.where(np.isfinite(b), np.maximum(b, 1.0), 1.0)
    return b.astype(np.int64) if b.ndim else int(b)


def certificate_ratio(certs: np.ndarray, errs: np.ndarray) -> np.ndarray:
    certs = np.asarray(certs, np.float64)
    errs = np.asarray(errs, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = certs / errs
    r[~np.isfinite(r)] = np.nan
    return r


def worst_case_ratio(rho: float) -> float:
    return (1.0 + rho) / (1.0 - rho)


def percentiles(a: np.ndarray, qs=(50, 90, 99, 100)) -> dict:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}
