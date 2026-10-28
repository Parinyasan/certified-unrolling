from __future__ import annotations

import numpy as np


class LinearOperator:

    shape: tuple[int, int]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dense(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def n(self) -> int:
        return self.shape[1]

    def _svals(self) -> np.ndarray:
        if not hasattr(self, "_svals_cache"):
            A = np.asarray(self.to_dense(), dtype=np.float64)
            self._svals_cache = np.linalg.svd(A, compute_uv=False)
        return self._svals_cache

    def sigma_max(self) -> float:
        return float(self._svals()[0])

    def sigma_min(self) -> float:
        if self.m < self.n:
            return 0.0
        return float(self._svals()[-1])

    def gram_spectral_norm(self, mu: float) -> float:
        s2 = self._svals() ** 2
        val = float(np.abs(1.0 - mu * s2).max())
        if self.m < self.n:
            val = max(val, 1.0)
        return val

    def gram(self) -> np.ndarray:
        A = self.to_dense()
        return A.T @ A


class DenseOperator(LinearOperator):

    def __init__(self, A: np.ndarray):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A must be 2-D")
        self.A = np.ascontiguousarray(A)
        self.shape = A.shape

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x @ self.A.T

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return y @ self.A

    def to_dense(self) -> np.ndarray:
        return self.A

    def __repr__(self) -> str:
        return f"DenseOperator(shape={self.shape})"


class IdentityOperator(LinearOperator):

    def __init__(self, n: int):
        self.shape = (n, n)

    def matvec(self, x):
        return np.asarray(x, dtype=np.float64)

    def rmatvec(self, y):
        return np.asarray(y, dtype=np.float64)

    def to_dense(self):
        return np.eye(self.n)

    def sigma_max(self):
        return 1.0

    def sigma_min(self):
        return 1.0


def as_operator(A) -> LinearOperator:
    if isinstance(A, LinearOperator):
        return A
    return DenseOperator(np.asarray(A))


class SubsampledOperator(LinearOperator):

    def __init__(self, rows: np.ndarray, n: int, base: LinearOperator | None = None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.base = base
        n_base = n if base is None else base.n
        self.shape = (len(self.rows), n_base)
        self._m_full = n if base is None else base.m

    def matvec(self, x):
        z = np.asarray(x, dtype=np.float64) if self.base is None else self.base.matvec(x)
        return z[..., self.rows]

    def rmatvec(self, y):
        y = np.asarray(y, dtype=np.float64)
        z = np.zeros(y.shape[:-1] + (self._m_full,), dtype=np.float64)
        z[..., self.rows] = y
        return z if self.base is None else self.base.rmatvec(z)

    def to_dense(self):
        base = np.eye(self._m_full) if self.base is None else self.base.to_dense()
        return np.ascontiguousarray(base[self.rows])


class ConvOperator(LinearOperator):

    def __init__(self, h: np.ndarray, n: int, center: int | None = None):
        h = np.asarray(h, dtype=np.float64)
        self.h = h
        c = (len(h) - 1) // 2 if center is None else center
        self.center = c
        A = np.zeros((n, n))
        idx = np.arange(n)
        for k, hk in enumerate(h):
            cols = idx - k + c
            ok = (cols >= 0) & (cols < n)
            A[idx[ok], cols[ok]] = hk
        self.A = A
        self.shape = (n, n)

    def matvec(self, x):
        return np.asarray(x, dtype=np.float64) @ self.A.T

    def rmatvec(self, y):
        return np.asarray(y, dtype=np.float64) @ self.A

    def to_dense(self):
        return self.A


def compose(M: LinearOperator, Psi: LinearOperator | np.ndarray) -> DenseOperator:
    Psi_d = Psi if isinstance(Psi, np.ndarray) else Psi.to_dense()
    return DenseOperator(M.to_dense() @ Psi_d)
