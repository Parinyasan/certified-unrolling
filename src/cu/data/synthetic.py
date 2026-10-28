from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cu.ops.linear import DenseOperator, LinearOperator


def gaussian_operator(m: int, n: int, rng: np.random.Generator, col_norm: float | None = 1.0) -> DenseOperator:
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    if col_norm is not None:
        A = A / np.linalg.norm(A, axis=0, keepdims=True) * col_norm
    return DenseOperator(A)


def conditioned_operator(n: int, kappa: float, rng: np.random.Generator, m: int | None = None) -> DenseOperator:
    m = n if m is None else m
    if m < n:
        raise ValueError("conditioned_operator needs m >= n (full column rank)")
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(-0.5 * np.log10(kappa), 0.5 * np.log10(kappa), n)
    return DenseOperator(U @ np.diag(s) @ V.T)


def sparse_signals(B: int, n: int, p: float, rng: np.random.Generator, amp_std: float = 1.0) -> np.ndarray:
    support = rng.random((B, n)) < p
    amp = rng.standard_normal((B, n)) * amp_std
    return support * amp


def add_noise(Y_clean: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None or np.isinf(snr_db):
        return Y_clean.copy()
    m = Y_clean.shape[-1]
    sig = np.linalg.norm(Y_clean, axis=-1, keepdims=True) / np.sqrt(m) * 10 ** (-snr_db / 20)
    return Y_clean + sig * rng.standard_normal(Y_clean.shape)


def lam_from_y(A: LinearOperator, Y: np.ndarray, lam_rel: float) -> np.ndarray:
    return lam_rel * np.max(np.abs(A.rmatvec(Y)), axis=-1)


@dataclass
class SynthBatch:
    A: LinearOperator
    Y: np.ndarray
    X_star: np.ndarray
    lam: np.ndarray
    eps: float
    meta: dict


def make_synth(
    B: int,
    m: int = 250,
    n: int = 500,
    p: float = 0.1,
    snr_db: float | None = 30.0,
    lam_rel: float = 0.1,
    eps_rel: float = 1e-2,
    seed: int = 0,
    A: LinearOperator | None = None,
    amp_std: float = 1.0,
) -> SynthBatch:
    rng = np.random.default_rng(seed)
    if A is None:
        A = gaussian_operator(m, n, rng)
    X = sparse_signals(B, A.n, p, rng, amp_std)
    Y = add_noise(A.matvec(X), snr_db, rng)
    lam = lam_from_y(A, Y, lam_rel)
    eps = eps_rel * A.sigma_max() ** 2
    meta = dict(m=A.m, n=A.n, p=p, snr_db=snr_db, lam_rel=lam_rel, eps_rel=eps_rel, seed=seed, B=B)
    return SynthBatch(A=A, Y=Y, X_star=X, lam=lam, eps=eps, meta=meta)


def load_synthetic(problem_cfg: dict, n_samples: int, seed: int, A: LinearOperator | None = None):
    p = problem_cfg
    rng = np.random.default_rng(seed)
    if A is None:
        if p["A"]["type"] == "gaussian":
            A = gaussian_operator(p["m"], p["n"], rng, col_norm=p["A"].get("col_norm", 1.0))
        elif p["A"]["type"] == "conditioned":
            A = conditioned_operator(p["n"], p["A"]["kappa"], rng, m=p["m"])
        else:
            raise ValueError(p["A"]["type"])
    else:
        rng = np.random.default_rng(seed + 12345)
    b = make_synth(n_samples, p["m"], p["n"], p["sparsity_p"], p.get("snr_db"), p["lam"], p["eps_rel"],
                   seed=int(rng.integers(0, 2**31 - 1)), A=A, amp_std=p.get("amplitude", {}).get("std", 1.0))
    meta = dict(b.meta, lam=b.lam, eps=b.eps, source="synthetic")
    return b.A, b.Y, b.X_star, meta
