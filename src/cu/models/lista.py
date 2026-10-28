from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cu.algos.prox import soft_threshold_t


def _inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x))) if x < 20 else x


class UnrolledHead(nn.Module):
    def __init__(self, A: np.ndarray, K: int, mu: float | None = None, dtype=torch.float32):
        super().__init__()
        A = np.asarray(A, dtype=np.float64)
        self.m, self.n = A.shape
        self.K = K
        smax = np.linalg.svd(A, compute_uv=False)[0]
        self.mu = float(1.0 / smax**2) if mu is None else float(mu)
        self.register_buffer("A", torch.as_tensor(A, dtype=dtype))
        self.t_raw = nn.Parameter(torch.full((K,), _inv_softplus(self.mu), dtype=dtype))

    def set_A(self, A: np.ndarray):
        self.A = torch.as_tensor(np.asarray(A), dtype=self.A.dtype, device=self.A.device)

    def theta(self, k: int, lam: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.t_raw[k]) * lam

    def layer_params(self, k: int) -> list[nn.Parameter]:
        raise NotImplementedError

    def step(self, k: int, x: torch.Tensor, y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, y: torch.Tensor, lam, n_layers: int | None = None, return_all: bool = False):
        y = torch.as_tensor(y, dtype=self.A.dtype, device=self.A.device)
        lam = torch.as_tensor(np.asarray(lam), dtype=self.A.dtype, device=self.A.device).reshape(-1, 1)
        if lam.shape[0] == 1 and y.shape[0] > 1:
            lam = lam.expand(y.shape[0], 1)
        n_layers = self.K if n_layers is None else n_layers
        x = torch.zeros(y.shape[0], self.n, dtype=y.dtype, device=y.device)
        xs = []
        for k in range(n_layers):
            x = self.step(k, x, y, lam)
            if return_all:
                xs.append(x)
        return xs if return_all else x


class LISTACP(UnrolledHead):

    def __init__(self, A, K, mu=None, dtype=torch.float32):
        super().__init__(A, K, mu, dtype)
        self.W = nn.ParameterList([nn.Parameter(self.mu * self.A.clone()) for _ in range(K)])

    def layer_params(self, k):
        return [self.W[k], self.t_raw]

    def step(self, k, x, y, lam):
        r = y - x @ self.A.T
        return soft_threshold_t(x + r @ self.W[k], self.theta(k, lam))


class TiedLISTACP(UnrolledHead):

    def __init__(self, A, K, mu=None, dtype=torch.float32):
        super().__init__(A, K, mu, dtype)
        self.W = nn.Parameter(self.mu * self.A.clone())

    def layer_params(self, k):
        return [self.W, self.t_raw]

    def step(self, k, x, y, lam):
        r = y - x @ self.A.T
        return soft_threshold_t(x + r @ self.W, self.theta(k, lam))


class LISTACPSS(LISTACP):

    def __init__(self, A, K, mu=None, dtype=torch.float32, p: float = 1.2, p_max: float = 13.0):
        super().__init__(A, K, mu, dtype)
        self.p, self.p_max = p, p_max

    def n_select(self, k: int) -> int:
        return int(round(min(self.p * (k + 1), self.p_max) / 100.0 * self.n))

    def step(self, k, x, y, lam):
        r = y - x @ self.A.T
        v = x + r @ self.W[k]
        out = soft_threshold_t(v, self.theta(k, lam))
        ns = self.n_select(k)
        if ns > 0:
            idx = torch.topk(v.abs(), ns, dim=1).indices
            out = out.scatter(1, idx, torch.gather(v, 1, idx))
        return out


class ALISTA(UnrolledHead):

    def __init__(self, A, K, mu=None, dtype=torch.float32, W: np.ndarray | None = None, n_iter_W: int = 500):
        super().__init__(A, K, mu, dtype)
        if W is None:
            W = analytic_weight(np.asarray(A, dtype=np.float64), n_iter=n_iter_W)
        self.register_buffer("W", torch.as_tensor(W, dtype=dtype))
        self.gamma_raw = nn.Parameter(torch.full((K,), _inv_softplus(1.0), dtype=dtype))

    def layer_params(self, k):
        return [self.gamma_raw, self.t_raw]

    def step(self, k, x, y, lam):
        r = y - x @ self.A.T
        return soft_threshold_t(x + F.softplus(self.gamma_raw[k]) * (r @ self.W), self.theta(k, lam))


class ConvLISTA(UnrolledHead):

    def __init__(self, A, K, mu=None, dtype=torch.float32, h: np.ndarray | None = None, P: int | None = None):
        super().__init__(A, K, mu, dtype)
        if h is None:
            Ad = np.asarray(A, dtype=np.float64)
            col = Ad[:, 0]
            nz = np.flatnonzero(np.abs(col) > 0)
            h = col[: nz[-1] + 1] if nz.size else col[:1]
        h = np.asarray(h, dtype=np.float64)
        self.P = int(P if P is not None else len(h) - 1)
        g0 = np.zeros(2 * self.P + 1)
        g0[self.P: self.P + len(h)] = self.mu * h
        self.g = nn.ParameterList([nn.Parameter(torch.as_tensor(g0.copy(), dtype=dtype)) for _ in range(K)])

    def layer_params(self, k):
        return [self.g[k], self.t_raw]

    def _corr(self, r: torch.Tensor, k: int) -> torch.Tensor:
        return F.conv1d(r.unsqueeze(1), self.g[k].view(1, 1, -1), padding=self.P).squeeze(1)

    def step(self, k, x, y, lam):
        r = y - x @ self.A.T
        return soft_threshold_t(x + self._corr(r, k), self.theta(k, lam))

    def dense_W(self, k: int) -> np.ndarray:
        g = self.g[k].detach().cpu().numpy().astype(np.float64)
        m, n = self.m, self.n
        W = np.zeros((m, n))
        idx = np.arange(n)
        for t in range(-self.P, self.P + 1):
            rows = idx + t
            ok = (rows >= 0) & (rows < m)
            W[rows[ok], idx[ok]] = g[t + self.P]
        return W


def analytic_weight(A: np.ndarray, n_iter: int = 500, lr: float | None = None) -> np.ndarray:
    m, n = A.shape
    col_norm2 = np.sum(A * A, axis=0)
    W = A / col_norm2
    G = A @ A.T
    lr = 1.0 / (2 * np.linalg.eigvalsh(G)[-1]) if lr is None else lr
    for _ in range(n_iter):
        W = W - lr * 2 * G @ W
        W = W + A * ((1.0 - np.sum(A * W, axis=0)) / col_norm2)
    return W


HEADS = dict(lista_cp=LISTACP, lista_cpss=LISTACPSS, alista=ALISTA, tied_lista_cp=TiedLISTACP, conv_lista=ConvLISTA)


def make_head(kind: str, A: np.ndarray, K: int, mu=None, dtype=torch.float32, **kw) -> UnrolledHead:
    return HEADS[kind](A, K, mu=mu, dtype=dtype, **kw)
