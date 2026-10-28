from __future__ import annotations

import numpy as np

from cu.algos.prox import elastic_net_prox
from cu.ops.linear import LinearOperator, as_operator


def _lam_col(lam):
    lam = np.asarray(lam, dtype=np.float64)
    if lam.ndim == 1:
        lam = lam[:, None]
    return lam


def objective(A: LinearOperator, y: np.ndarray, x: np.ndarray, lam, eps=0.0) -> np.ndarray:
    A = as_operator(A)
    r = A.matvec(x) - y
    lam = _lam_col(lam)
    return (
        0.5 * np.sum(r * r, axis=-1)
        + np.sum(lam * np.abs(x), axis=-1)
        + 0.5 * eps * np.sum(x * x, axis=-1)
    )


def default_step(A: LinearOperator) -> float:
    return 1.0 / as_operator(A).sigma_max() ** 2


def _prep(A, y, lam, x0):
    A = as_operator(A)
    y = np.asarray(y, dtype=np.float64)
    lam = _lam_col(lam)
    if x0 is None:
        x = np.zeros(y.shape[:-1] + (A.n,), dtype=np.float64)
    else:
        x = np.array(x0, dtype=np.float64, copy=True)
    return A, y, lam, x


def ista(A, y, lam, n_iter: int, mu: float | None = None, x0=None, eps: float = 0.0, return_trace=False):
    A, y, lam, x = _prep(A, y, lam, x0)
    mu = default_step(A) if mu is None else float(mu)
    trace = [x.copy()] if return_trace else None
    for _ in range(n_iter):
        g = A.rmatvec(A.matvec(x) - y)
        x = elastic_net_prox(x - mu * g, mu * lam, mu * eps)
        if return_trace:
            trace.append(x.copy())
    return (x, trace) if return_trace else x


def elastic_net_ista(A, y, lam, eps, n_iter: int, mu: float | None = None, x0=None, return_trace=False):
    return ista(A, y, lam, n_iter, mu=mu, x0=x0, eps=eps, return_trace=return_trace)


def fista(A, y, lam, n_iter: int, mu: float | None = None, x0=None, eps: float = 0.0, return_trace=False):
    A, y, lam, x = _prep(A, y, lam, x0)
    mu = default_step(A) if mu is None else float(mu)
    z = x.copy()
    t = 1.0
    trace = [x.copy()] if return_trace else None
    for _ in range(n_iter):
        g = A.rmatvec(A.matvec(z) - y)
        x_new = elastic_net_prox(z - mu * g, mu * lam, mu * eps)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if return_trace:
            trace.append(x.copy())
    return (x, trace) if return_trace else x


__all__ = ["objective", "default_step", "ista", "fista", "elastic_net_ista"]
