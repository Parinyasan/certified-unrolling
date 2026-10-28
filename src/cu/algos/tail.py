from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from cu.algos.prox import elastic_net_prox
from cu.ops.linear import LinearOperator, as_operator


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))


@dataclass
class RunResult:

    x: np.ndarray
    k_star: np.ndarray | int
    certs: np.ndarray
    diffs: np.ndarray
    errs_vs_xhat: np.ndarray | None
    time_s: float
    converged: np.ndarray | bool
    rho: float
    eps_tol: np.ndarray | float
    extra: dict = field(default_factory=dict)


class ContractiveTail:

    def __init__(self, A: LinearOperator, lam, eps: float, mu: float | None = None):
        self.A = as_operator(A)
        self.lam = np.asarray(lam, dtype=np.float64)
        self.eps = float(eps)
        if self.eps < 0:
            raise ValueError("eps must be >= 0")
        self.smax = self.A.sigma_max()
        self.smin = self.A.sigma_min()
        if mu is None:
            mu = self.default_mu()
        self.mu = float(mu)
        if self.mu <= 0:
            raise ValueError("mu must be > 0")

    def default_mu(self) -> float:
        return 2.0 / (self.smin**2 + self.smax**2)

    def lipschitz_G(self) -> float:
        return max(abs(1.0 - self.mu * self.smin**2), abs(1.0 - self.mu * self.smax**2))

    def rho(self) -> float:
        return self.lipschitz_G() / (1.0 + self.mu * self.eps)

    def is_contraction(self) -> bool:
        return self.rho() < 1.0

    def certificate_factor(self) -> float:
        rho = self.rho()
        if rho >= 1.0:
            return math.inf
        return rho / (1.0 - rho)

    def _lam_b(self, lam):
        lam = self.lam if lam is None else np.asarray(lam, dtype=np.float64)
        if lam.ndim == 1:
            lam = lam[:, None]
        return lam

    def step(self, x: np.ndarray, y: np.ndarray, lam=None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        g = self.A.rmatvec(self.A.matvec(x) - y)
        return elastic_net_prox(x - self.mu * g, self.mu * self._lam_b(lam), self.mu * self.eps)

    def certificate(self, x_k: np.ndarray, x_km1: np.ndarray):
        return self.certificate_factor() * _norm(np.asarray(x_k, np.float64) - np.asarray(x_km1, np.float64))

    def run_until(
        self,
        x0: np.ndarray | None,
        y: np.ndarray,
        eps_tol,
        max_iter: int = 100_000,
        xhat: np.ndarray | None = None,
        lam=None,
        record: bool = True,
        run_all: bool = False,
        rel_tau: float | None = None,
    ) -> RunResult:
        if not self.is_contraction():
            raise ValueError(f"rho = {self.rho():.6g} >= 1: the tail is not a contraction; no certificate")
        y = np.asarray(y, dtype=np.float64)
        batch = y.ndim == 2
        if x0 is None:
            x = np.zeros(y.shape[:-1] + (self.A.n,), dtype=np.float64)
        else:
            x = np.array(x0, dtype=np.float64, copy=True)
        B = y.shape[0] if batch else 1
        eps_tol_arr = np.broadcast_to(np.asarray(eps_tol, dtype=np.float64), (B,)).copy()
        lam_b = self._lam_b(lam)
        fac = self.certificate_factor()
        if xhat is not None:
            xhat = np.asarray(xhat, dtype=np.float64)

        done = np.zeros(B, dtype=bool)
        k_star = np.full(B, -1, dtype=np.int64)
        x_out = x.copy()
        certs, diffs, errs = [], [], []

        t0 = time.perf_counter()
        k = 0
        while k < max_iter:
            k += 1
            x_new = self.step(x, y, lam_b)
            d = np.atleast_1d(_norm(x_new - x))
            c = fac * d
            if record:
                diffs.append(d)
                certs.append(c)
                if xhat is not None:
                    errs.append(np.atleast_1d(_norm(x_new - xhat)))
            if rel_tau is None:
                fired = (~done) & (c <= eps_tol_arr)
            else:
                fired = (~done) & (c <= rel_tau * (np.atleast_1d(_norm(x_new)) - c))
            if fired.any():
                k_star[fired] = k
                if batch:
                    x_out[fired] = x_new[fired]
                else:
                    x_out = x_new.copy()
                done |= fired
            x = x_new
            if done.all() and not run_all:
                break
        time_s = time.perf_counter() - t0

        not_done = ~done
        if not_done.any():
            if batch:
                x_out[not_done] = x[not_done]
            else:
                x_out = x.copy()
            k_star[not_done] = k

        def _stack(lst):
            if not record or not lst:
                return None
            arr = np.stack(lst, axis=-1)
            return arr if batch else arr[0]

        return RunResult(
            x=x_out,
            k_star=k_star if batch else int(k_star[0]),
            certs=_stack(certs),
            diffs=_stack(diffs),
            errs_vs_xhat=_stack(errs) if xhat is not None else None,
            time_s=time_s,
            converged=done if batch else bool(done[0]),
            rho=self.rho(),
            eps_tol=eps_tol_arr if batch else float(eps_tol_arr[0]),
        )

    def solve_fixed_point(self, y, lam=None, tol: float = 1e-12, max_iter: int = 100_000, x0=None):
        y = np.asarray(y, dtype=np.float64)
        batch = y.ndim == 2
        Y = y if batch else y[None, :]
        lam_b = self._lam_b(lam)
        if lam_b.ndim == 0:
            lam_b = np.broadcast_to(lam_b, (Y.shape[0], 1))
        X = np.zeros((Y.shape[0], self.A.n), dtype=np.float64) if x0 is None else np.array(x0, np.float64, copy=True).reshape(Y.shape[0], -1)
        active = np.ones(Y.shape[0], dtype=bool)
        n_iter = np.zeros(Y.shape[0], dtype=np.int64)
        last_diff = np.full(Y.shape[0], np.inf)
        k = 0
        while active.any() and k < max_iter:
            k += 1
            idx = np.flatnonzero(active)
            Xa = X[idx]
            Xn = self.step(Xa, Y[idx], lam_b[idx])
            d = _norm(Xn - Xa)
            X[idx] = Xn
            last_diff[idx] = d
            n_iter[idx] = k
            active[idx[d <= tol]] = False
        return (X if batch else X[0]), n_iter, last_diff

    def __repr__(self) -> str:
        return (f"ContractiveTail(shape={self.A.shape}, eps={self.eps:.4g}, mu={self.mu:.4g}, "
                f"smax={self.smax:.4g}, smin={self.smin:.4g}, rho={self.rho():.6f})")


def certified_solve(head, tail: ContractiveTail, y, eps_tol, max_iter: int = 100_000, xhat=None, lam=None) -> RunResult:
    if head is None:
        x0 = None
    else:
        import torch

        with torch.no_grad():
            y_t = torch.as_tensor(np.asarray(y), dtype=next(head.parameters()).dtype)
            x0 = head(y_t).detach().double().cpu().numpy()
    return tail.run_until(x0, y, eps_tol, max_iter=max_iter, xhat=xhat, lam=lam)
