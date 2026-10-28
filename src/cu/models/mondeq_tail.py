from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cu.algos.tail import ContractiveTail
from cu.ops.linear import IdentityOperator


class MonDEQTail(nn.Module):
    def __init__(self, m_y: int, n_out: int, d: int | None = None, m: float = 0.5, init_scale: float = 0.05, dtype=torch.float32):
        super().__init__()
        d = n_out if d is None else d
        self.d, self.m_y, self.n_out, self.m = d, m_y, n_out, float(m)
        self.P = nn.Parameter(init_scale * torch.randn(d, d, dtype=dtype) / d**0.5)
        self.Q = nn.Parameter(init_scale * torch.randn(d, d, dtype=dtype) / d**0.5)
        self.U = nn.Parameter(torch.randn(d, m_y, dtype=dtype) / m_y**0.5)
        self.b = nn.Parameter(torch.zeros(d, dtype=dtype))
        self.C = nn.Parameter(torch.randn(n_out, d, dtype=dtype) / d**0.5)
        self.D = nn.Parameter(torch.zeros(n_out, m_y, dtype=dtype))

    def I_minus_W(self) -> torch.Tensor:
        d = self.d
        return self.m * torch.eye(d, dtype=self.P.dtype) + self.P.T @ self.P - (self.Q - self.Q.T)

    def lipschitz(self, n_iter: int = 30, exact: bool = False) -> torch.Tensor:
        M = self.I_minus_W()
        if exact:
            return torch.linalg.matrix_norm(M, ord=2)
        v = torch.randn(self.d, 1, dtype=M.dtype)
        MtM = M.T @ M
        with torch.no_grad():
            for _ in range(n_iter):
                v = MtM @ v
                v = v / v.norm()
        return torch.sqrt((v.T @ (MtM @ v)).squeeze() / (v.T @ v).squeeze())

    def alpha(self, L: torch.Tensor | float) -> float:
        return float(self.m / float(L) ** 2)

    def rho_B(self, L=None, alpha=None) -> float:
        L = float(self.lipschitz(exact=True)) if L is None else float(L)
        a = self.alpha(L) if alpha is None else float(alpha)
        return float(np.sqrt(max(1.0 - 2 * a * self.m + a * a * L * L, 0.0)))

    def fb_step(self, z, y, M, alpha):
        return F.relu(z - alpha * (z @ M.T - y @ self.U.T - self.b))

    def readout(self, z, y):
        return z @ self.C.T + y @ self.D.T

    def forward(self, y, z0=None, max_iter: int = 200, tol: float = 1e-4, return_k: bool = False):
        M = self.I_minus_W()
        L = self.lipschitz(exact=False)
        alpha = self.alpha(L.detach())
        z = torch.zeros(y.shape[0], self.d, dtype=y.dtype) if z0 is None else z0
        with torch.no_grad():
            k = 0
            for k in range(1, max_iter + 1):
                zn = self.fb_step(z, y, M.detach(), alpha)
                d = (zn - z).norm(dim=1).max().item()
                z = zn
                if d <= tol:
                    break
        z = self.fb_step(z, y, M, alpha)
        x = self.readout(z, y)
        return (x, z, k) if return_k else x

    def numpy_tail(self) -> "MonDEQNumpyTail":
        with torch.no_grad():
            M = self.I_minus_W().double().numpy()
            return MonDEQNumpyTail(M, self.U.double().numpy(), self.b.double().numpy(), self.m,
                                   self.C.double().numpy(), self.D.double().numpy())


class MonDEQNumpyTail(ContractiveTail):

    def __init__(self, M: np.ndarray, U: np.ndarray, b: np.ndarray, m: float, C: np.ndarray, D: np.ndarray, alpha: float | None = None):
        self.M = np.asarray(M, np.float64); self.U = np.asarray(U, np.float64); self.b = np.asarray(b, np.float64)
        self.C = np.asarray(C, np.float64); self.D = np.asarray(D, np.float64)
        self.m = float(m)
        self.L = float(np.linalg.norm(self.M, 2))
        self.alpha_bound_ = self.m / self.L**2
        self.alpha_ = self._alpha_exact() if alpha is None else float(alpha)
        self.rho_exact_ = float(np.linalg.norm(np.eye(self.M.shape[0]) - self.alpha_ * self.M, 2))
        self.A = IdentityOperator(self.M.shape[0]); self.lam = np.float64(0.0); self.eps = 0.0; self.mu = 1.0
        self.smax = self.smin = 1.0
        self.C_norm = float(np.linalg.norm(self.C, 2))

    def _alpha_exact(self) -> float:
        I = np.eye(self.M.shape[0])
        f = lambda a: np.linalg.norm(I - a * self.M, 2)
        lo, hi = 1e-6 * self.alpha_bound_, 2 * self.m / self.L**2
        g = (np.sqrt(5) - 1) / 2
        a, b = hi - g * (hi - lo), lo + g * (hi - lo); fa, fb = f(a), f(b)
        for _ in range(40):
            if fa < fb:
                hi, b, fb = b, a, fa; a = hi - g * (hi - lo); fa = f(a)
            else:
                lo, a, fa = a, b, fb; b = lo + g * (hi - lo); fb = f(b)
        return float(0.5 * (lo + hi))

    def rho_bound(self) -> float:
        a = self.alpha_
        return float(np.sqrt(max(1.0 - 2 * a * self.m + a * a * self.L**2, 0.0)))

    def strong_monotonicity_gap(self) -> float:
        return float(np.linalg.eigvalsh(0.5 * (self.M + self.M.T))[0] - self.m)

    def rho(self) -> float:
        return self.rho_exact_

    def step(self, z, y, lam=None):
        z = np.asarray(z, np.float64); y = np.asarray(y, np.float64)
        return np.maximum(z - self.alpha_ * (z @ self.M.T - y @ self.U.T - self.b), 0.0)

    def readout(self, z, y):
        return np.asarray(z, np.float64) @ self.C.T + np.asarray(y, np.float64) @ self.D.T


class ZEncoder(nn.Module):

    def __init__(self, m_y: int, d: int, dtype=torch.float32):
        super().__init__()
        self.lin = nn.Linear(m_y, d, dtype=dtype)
        nn.init.zeros_(self.lin.bias)

    def forward(self, y):
        return F.relu(self.lin(y))
