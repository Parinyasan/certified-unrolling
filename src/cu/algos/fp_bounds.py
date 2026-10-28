from __future__ import annotations

import math

import numpy as np

U64 = 2.0 ** -53
U32 = 2.0 ** -24


def gamma(k: int, u: float = U64) -> float:
    return k * u / (1.0 - k * u)


def _soft(v, th):
    return np.sign(v) * np.maximum(np.abs(v) - th, 0.0)


def step_error_bound(Ad, absA, Y, X, lam, eps, mu, u=U64, N_A=None):
    m, n = Ad.shape
    th = mu * np.asarray(lam, np.float64).reshape(-1, 1)
    a = np.abs(X) @ absA.T + np.abs(Y)
    bg = gamma(m + n + 2, u) * (a @ absA)
    r = X @ Ad.T - Y
    g = r @ Ad
    v = X - mu * g
    c = mu * eps
    bv = mu * bg + gamma(2, u) * mu * np.abs(g) + gamma(1, u) * np.abs(v)
    bth = gamma(2, u) * th
    infl = (1 + 8 * U64) / (1 - gamma(m + 2 * n + 10, U64))
    act = (np.abs(v) + (bv + bth) * infl) * (1 + 2 * U64) > th
    s = _soft(v, th)
    t = s / (1.0 + c)
    b = act * (bv + bth + gamma(1, u) * np.abs(s)) / (1.0 + c) + gamma(6, u) * np.abs(t)
    comp = np.linalg.norm(b, axis=1) * infl
    if N_A is None:
        return comp, None
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    normw = ((mu * gamma(m + n + 2, u) * (N_A ** 2 * nx + N_A * ny) + gamma(2, u) * mu * np.linalg.norm(g, axis=1)
              + gamma(1, u) * np.linalg.norm(v, axis=1) + bth[:, 0] * math.sqrt(n) + gamma(1, u) * np.linalg.norm(s, axis=1)) / (1.0 + c)
             + gamma(6, u) * np.linalg.norm(t, axis=1))
    return comp, normw * infl


def gap_eval_bound(Ad, absA, Y, X, lam, eps, u=U64, N_A=None):
    m, n = Ad.shape
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    r = X @ Ad.T - Y
    g = r @ Ad
    P = 0.5 * np.sum(r * r, 1) + np.sum(lamc * np.abs(X), 1) + 0.5 * eps * np.sum(X * X, 1)
    sg = _soft(g, lamc)
    D = -np.sum(r * Y, 1) - 0.5 * np.sum(r * r, 1) - np.sum(sg ** 2, 1) / (2 * eps)
    nr = np.linalg.norm(r, axis=1)
    nsg = np.linalg.norm(sg, axis=1)
    l1 = np.sum(lamc * np.abs(X), 1)
    x2 = np.sum(X * X, 1)
    ry = np.sum(np.abs(r * Y), 1)
    ry_abs = np.abs(np.sum(r * Y, 1))

    def assemble(ndr, ndg):
        EP = nr * ndr + 0.5 * ndr ** 2 + gamma(m + 2, u) * 0.5 * nr ** 2 + gamma(n + 2, u) * l1 + gamma(n + 3, u) * 0.5 * eps * x2
        ED = (gamma(m + 1, u) * ry + gamma(m + 2, u) * 0.5 * nr ** 2
              + ((2 * nsg + ndg) * ndg + gamma(n + 3, u) * nsg ** 2 + u * nsg ** 2) / (2 * eps))
        Sigma = P + ry_abs + 0.5 * nr ** 2 + nsg ** 2 / (2 * eps)
        return ((EP + ED) * (1 + gamma(4, u)) + gamma(3, u) * Sigma) * (1 + 8 * U64) / (1 - gamma(m + 2 * n + 10, U64))

    dr = gamma(n + 1, u) * (np.abs(X) @ absA.T + np.abs(Y))
    dg = gamma(m, u) * (np.abs(r) @ absA)
    act = (np.abs(g) + dg * (1 + 8 * U64) / (1 - gamma(m + 2 * n + 10, U64))) * (1 + 2 * U64) > lamc
    E = assemble(np.linalg.norm(dr, axis=1), np.linalg.norm(dg * act, axis=1))
    En = None
    if N_A is not None:
        En = assemble(gamma(n + 1, u) * (N_A * np.linalg.norm(X, axis=1) + np.linalg.norm(Y, axis=1)), gamma(m, u) * N_A * nr)
    return P, D, E, En


def abs_norm_bound(absA):
    m = absA.shape[0]
    return math.sqrt(np.max(absA @ (absA.T @ np.ones(m)))) * (1 + 8 * U64)


def verified_sigma(Ad, lower=False):
    u = U64
    m, n = Ad.shape
    A_ = Ad if m <= n else Ad.T
    p, q = A_.shape
    absA = np.abs(A_)
    M = A_ @ A_.T
    M = np.tril(M) + np.tril(M, -1).T
    dM2 = gamma(q, u) * np.max((absA @ absA.T).sum(1)) * (1 + gamma(q + 2, u))
    w, V = np.linalg.eigh(M)
    B = V.T @ (M @ V)
    EB = gamma(2 * p + 2, u) * (np.abs(V).T @ (np.abs(M) @ np.abs(V))) * (1 + gamma(2 * p + 4, u))
    dB = np.diag(B)
    off = (np.abs(B) * (1.0 - np.eye(p))).sum(1)
    rowE = EB.sum(1)
    slack = gamma(p + 3, u) * (np.abs(dB) + off + rowE)
    G = V.T @ V
    EG = gamma(p + 1, u) * (np.abs(V).T @ np.abs(V)) * (1 + gamma(p + 3, u))
    etaV = np.max(np.abs(G - np.eye(p)).sum(1) + EG.sum(1)) * (1 + gamma(p + 3, u))
    lmaxB = np.max(dB + off + rowE + slack)
    s2_up = (lmaxB / (1 - etaV) + dM2) * (1 + 8 * u)
    out = dict(sigma_bar=math.sqrt(s2_up) * (1 + 2 * u), eta_V=float(etaV), dM2=float(dM2))
    s = np.linalg.svd(Ad, compute_uv=False)
    out["svd_sigma_max"] = float(s[0])
    out["svd_sigma_min_gram"] = float(s[-1]) if m == n or m > n else 0.0
    out["sigma_low"] = None
    if lower and m >= n:
        lminB = np.min(dB - off - rowE - slack)
        l_low = (lminB / (1 + etaV) - dM2) * (1 - 8 * u) if lminB > 0 else 0.0
        out["sigma_low"] = math.sqrt(l_low) * (1 - 2 * u) if l_low > 0 else 0.0
    return out


def cheap_sigma_bounds(Ad):
    absA = np.abs(Ad)
    m, n = Ad.shape
    AAT = Ad @ Ad.T
    v = np.ones(m)
    for _ in range(50):
        Nv = absA @ (absA.T @ v)
        cw = math.sqrt(np.max(Nv / v))
        v = Nv / np.max(Nv)
    return {
        "aat_inf": math.sqrt(np.abs(AAT).sum(1).max()),
        "ata_inf": math.sqrt(np.abs(Ad.T @ Ad).sum(1).max()),
        "one_inf": math.sqrt(np.abs(Ad).sum(0).max() * np.abs(Ad).sum(1).max()),
        "collatz_abs50": cw,
        "frobenius": float(np.linalg.norm(Ad, "fro")),
    }


def power_estimate(Ad, k, rng):
    x = rng.standard_normal(Ad.shape[1])
    x /= np.linalg.norm(x)
    th = 0.0
    for _ in range(k):
        z = Ad.T @ (Ad @ x)
        th = float(x @ z)
        x = z / np.linalg.norm(z)
    return math.sqrt(th)


def lanczos_estimate(Ad, k, rng):
    q = rng.standard_normal(Ad.shape[1])
    q /= np.linalg.norm(q)
    Q = [q]
    al, be = [], []
    for j in range(k):
        z = Ad.T @ (Ad @ Q[-1])
        al.append(float(Q[-1] @ z))
        Qm = np.array(Q)
        for _ in range(2):
            z = z - Qm.T @ (Qm @ z)
        b = float(np.linalg.norm(z))
        if j < k - 1:
            if b < 1e-14:
                break
            be.append(b)
            Q.append(z / b)
    kk = len(al)
    T = np.diag(al) + np.diag(be[:kk - 1], 1) + np.diag(be[:kk - 1], -1)
    return math.sqrt(np.linalg.eigvalsh(T)[-1])


def kw_lanczos_inflation(n, k, fail):
    d = (math.log(1.648 * math.sqrt(n) / fail) / (2 * k - 1)) ** 2
    return float("inf") if d >= 1 else 1.0 / (1.0 - d)
