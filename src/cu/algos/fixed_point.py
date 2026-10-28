from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from cu.algos.tail import ContractiveTail
from cu.ops.linear import as_operator

DEFAULT_CACHE_DIR = Path(os.environ.get("CU_CACHE_DIR", "cache")) / "xhat"


def _hash_arrays(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:20]


def compute_and_cache_xhat(
    y,
    A,
    lam,
    eps: float,
    mu: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 100_000,
    cache_dir: str | os.PathLike | None = None,
    verbose: bool = False,
):
    A = as_operator(A)
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    tail = ContractiveTail(A, lam, eps, mu)
    if not tail.is_contraction():
        raise ValueError(f"rho={tail.rho():.6g} >= 1; fixed point of T_eps not certified unique")

    key = _hash_arrays(A.to_dense(), y, lam, float(eps), float(tail.mu), float(tol), int(max_iter))
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache_dir / f"xhat_{key}.npz"
    if path.exists():
        z = np.load(path)
        info = dict(n_iter=z["n_iter"], final_diff=z["final_diff"], cached=True, time_s=0.0, key=key, rho=tail.rho())
        return z["xhat"], info

    t0 = time.perf_counter()
    xhat, n_iter, final_diff = tail.solve_fixed_point(y, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    if verbose:
        print(f"[xhat] B={1 if y.ndim == 1 else y.shape[0]} rho={tail.rho():.6f} iters max={n_iter.max()} "
              f"mean={n_iter.mean():.1f} final_diff max={final_diff.max():.2e} time={dt:.2f}s")
    if (final_diff > tol).any():
        raise RuntimeError(f"xhat did not converge to tol={tol:g} within {max_iter} iterations "
                           f"(max final diff {final_diff.max():.3e}); refuse to cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, xhat=xhat, n_iter=n_iter, final_diff=final_diff, rho=tail.rho(), mu=tail.mu, eps=eps, tol=tol)
    info = dict(n_iter=n_iter, final_diff=final_diff, cached=False, time_s=dt, key=key, rho=tail.rho())
    return xhat, info
