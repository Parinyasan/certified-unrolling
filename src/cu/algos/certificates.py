from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cu.algos.prox import elastic_net_prox, soft_threshold
from cu.ops.linear import LinearOperator, as_operator


GAP_FLOOR_C = 16.0


def _lam_col(lam):
    lam = np.asarray(lam, np.float64)
    return lam[:, None] if lam.ndim == 1 else lam


def primal(A, y, x, lam, eps):
    A = as_operator(A)
    r = A.matvec(x) - y
    lam = _lam_col(lam)
    return 0.5 * np.sum(r * r, -1) + np.sum(lam * np.abs(x), -1) + 0.5 * eps * np.sum(x * x, -1)


def gap_certificate(A, y, x, lam, eps, r=None, ATr=None):
    A = as_operator(A)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    lam = _lam_col(lam)
    if r is None:
        r = A.matvec(x) - y
    if ATr is None:
        ATr = A.rmatvec(r)
    P = 0.5 * np.sum(r * r, -1) + np.sum(lam * np.abs(x), -1) + 0.5 * eps * np.sum(x * x, -1)
    u = r
    D = -np.sum(u * y, -1) - 0.5 * np.sum(u * u, -1) - np.sum(soft_threshold(ATr, lam) ** 2, -1) / (2 * eps)
    floor = GAP_FLOOR_C * np.finfo(np.float64).eps * (np.abs(P) + np.abs(D))
    gap = np.maximum(P - D, floor)
    return np.sqrt(2 * gap / eps), gap


def gap_safe_inactive(A, y, x, lam, eps, col_norms2=None, Ax=None, ATAx_minus_ATy=None):
    A = as_operator(A)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    lam = _lam_col(lam)
    if col_norms2 is None:
        Ad = A.to_dense(); col_norms2 = np.sum(Ad * Ad, 0)
    at_norm = np.sqrt(col_norms2 + eps)
    r = y - (A.matvec(x) if Ax is None else Ax)
    ATr_aug = (A.rmatvec(r) if ATAx_minus_ATy is None else -ATAx_minus_ATy) - eps * x
    P = 0.5 * np.sum(r * r, -1) + 0.5 * eps * np.sum(x * x, -1) + np.sum(lam * np.abs(x), -1)
    scale = np.maximum(lam[:, 0] if lam.ndim == 2 else lam, np.max(np.abs(ATr_aug), -1))
    lam_b = lam[:, 0] if lam.ndim == 2 else np.broadcast_to(lam, (x.shape[0],))
    theta_r = r / scale[:, None]; theta_x = -np.sqrt(eps) * x / scale[:, None]
    ny2 = np.sum(y * y, -1)
    D = 0.5 * ny2 - 0.5 * lam_b**2 * (np.sum((theta_r - y / lam_b[:, None]) ** 2, -1) + np.sum(theta_x**2, -1))
    gapL = np.maximum(P - D, GAP_FLOOR_C * np.finfo(np.float64).eps * (np.abs(P) + np.abs(D)))
    R = np.sqrt(2 * gapL) / lam_b
    corr = np.abs(ATr_aug) / scale[:, None]
    return corr + R[:, None] * at_norm[None, :] < 1.0, gapL


def rho_support(Ad: np.ndarray, S_mask: np.ndarray, mu: float, eps: float) -> float:
    S = np.flatnonzero(S_mask)
    if len(S) == 0:
        return 0.0
    w = np.linalg.eigvalsh(Ad[:, S].T @ Ad[:, S])
    return max(abs(1 - mu * w[0]), abs(1 - mu * w[-1])) / (1 + mu * eps)


@dataclass
class CertRun:
    k_star: np.ndarray
    x: np.ndarray
    time_s: float
    violations: int = 0
    checks: int = 0
    ratios: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def run_ista_certified(A, y, lam, eps, mu, x0, tau, xhat=None, mode="banach", max_iter=100_000, rho=None, cold_is_zero=True):
    A = as_operator(A); Ad = A.to_dense()
    y = np.asarray(y, np.float64); lam_b = _lam_col(lam)
    B, n = y.shape[0], A.n
    x = np.zeros((B, n)) if x0 is None else np.array(x0, np.float64, copy=True)
    if rho is None:
        s = np.linalg.svd(Ad, compute_uv=False); smin = 0.0 if A.m < A.n else s[-1]
        rho = max(abs(1 - mu * smin**2), abs(1 - mu * s[0] ** 2)) / (1 + mu * eps)
    fac = rho / (1 - rho)
    col2 = np.sum(Ad * Ad, 0)
    done = np.zeros(B, bool); k_star = np.full(B, -1); x_out = x.copy()
    viol = 0; checks = 0; ratios = []; k_S_equal = np.full(B, -1); rhoS_hist = []; rho_cache = {}
    rs_stop = np.full(B, np.nan); S_stop = np.full(B, -1)
    t0 = time.perf_counter()
    for k in range(1, max_iter + 1):
        r = A.matvec(x) - y
        g = A.rmatvec(r)
        xn = elastic_net_prox(x - mu * g, mu * lam_b, mu * eps)
        diff = np.linalg.norm(xn - x, axis=1)
        if mode == "banach":
            C = fac * diff
        elif mode == "gap":
            C, _ = gap_certificate(A, y, x, lam_b, eps, r=r, ATr=g)
        elif mode == "rhoS":
            inactive, _ = gap_safe_inactive(A, y, x, lam_b, eps, col2, Ax=r + y, ATAx_minus_ATy=g)
            S = (~inactive) | (x != 0) | (xn != 0)
            C = np.empty(B); rs_now = np.full(B, np.nan)
            for i in range(B):
                if done[i]:
                    C[i] = 0.0; continue
                if S[i].sum() > n // 2:
                    rs = rho
                else:
                    key = S[i].tobytes()
                    rs = rho_cache.get(key)
                    if rs is None:
                        rs = rho_support(Ad, S[i], mu, eps); rho_cache[key] = rs
                rs_now[i] = rs
                C[i] = (rs / (1 - rs)) * diff[i] if rs < 1 else fac * diff[i]
                if xhat is not None and k_S_equal[i] < 0 and np.array_equal(S[i], xhat[i] != 0):
                    k_S_equal[i] = k
            rhoS_hist.append(float(np.nanmedian(rs_now)) if np.isfinite(rs_now).any() else np.nan)
        else:
            raise ValueError(mode)
        cand = x if mode == "gap" else xn
        if xhat is not None:
            err = np.linalg.norm(cand - xhat, axis=1)
            act = ~done
            viol += int(np.sum((C < err - 1e-9) & act)); checks += int(act.sum())
            ok = act & (err > 0)
            ratios.append(C[ok] / err[ok])
        fired = (~done) & (C <= tau * (np.linalg.norm(cand, axis=1) - C))
        k_star[fired] = k; x_out[fired] = cand[fired]; done |= fired
        if mode == "rhoS" and fired.any():
            rs_stop[fired] = rs_now[fired]; S_stop[fired] = S[fired].sum(1)
        x = xn
        if done.all():
            break
    k_star[~done] = max_iter; x_out[~done] = x[~done]
    extra = {}
    if mode == "rhoS":
        extra["k_S_equals_supp_xhat"] = k_S_equal; extra["rhoS_median_trace"] = np.array(rhoS_hist)
        extra["rhoS_at_stop"] = rs_stop; extra["S_size_at_stop"] = S_stop
    return CertRun(k_star=k_star, x=x_out, time_s=time.perf_counter() - t0, violations=viol, checks=checks,
                   ratios=[np.concatenate(ratios)] if ratios else [], extra=extra)


def select_start(A, y, x0, lam, eps):
    A = as_operator(A); y = np.asarray(y, np.float64); lam_b = _lam_col(lam)
    x0 = np.asarray(x0, np.float64)
    r0 = A.matvec(x0) - y
    G_head, _ = gap_certificate(A, y, x0, lam_b, eps, r=r0, ATr=A.rmatvec(r0))
    G_zero, _ = gap_certificate(A, y, np.zeros_like(x0), lam_b, eps, r=-y, ATr=-A.rmatvec(y))
    keep = G_head <= G_zero
    return np.where(keep[:, None], x0, 0.0), keep, G_head, G_zero


def run_fista_gap(A, y, lam, eps, mu, x0, tau, xhat=None, max_iter=100_000, restart=True, refresh_every=50):
    A = as_operator(A); y = np.asarray(y, np.float64); lam_b = _lam_col(lam)
    B, n = y.shape[0], A.n
    mu = min(float(mu), 1.0 / A.sigma_max() ** 2)
    x = np.zeros((B, n)) if x0 is None else np.array(x0, np.float64, copy=True)
    z = x.copy(); t = np.ones(B)
    ATy = A.rmatvec(y)
    Ax_prev = np.zeros((B, A.m)); ATAx_prev = np.zeros((B, n)); beta_prev = np.zeros((B, 1))
    done = np.zeros(B, bool); k_star = np.full(B, -1); k_iter = np.full(B, -1); x_out = x.copy()
    viol = 0; checks = 0; ratios = []; extra_ops = 1
    t0 = time.perf_counter()
    for k in range(1, max_iter + 1):
        Az = A.matvec(z); gz = A.rmatvec(Az - y)
        if refresh_every and k % refresh_every == 0:
            Ax_c = A.matvec(x); ATAx_c = A.rmatvec(Ax_c); extra_ops += 1
        else:
            w = 1.0 / (1.0 + beta_prev)
            Ax_c = w * (Az + beta_prev * Ax_prev); ATAx_c = w * (gz + ATy + beta_prev * ATAx_prev)
        r_c = Ax_c - y
        C, _ = gap_certificate(A, y, x, lam_b, eps, r=r_c, ATr=ATAx_c - ATy)
        if xhat is not None:
            err = np.linalg.norm(x - xhat, axis=1); act = ~done
            viol += int(np.sum((C < err - 1e-9) & act)); checks += int(act.sum())
            ok = act & (err > 0); ratios.append(C[ok] / err[ok])
        fired = (~done) & (C <= tau * (np.linalg.norm(x, axis=1) - C))
        k_star[fired] = k + extra_ops; k_iter[fired] = k - 1; x_out[fired] = x[fired]; done |= fired
        if done.all():
            break
        xn = elastic_net_prox(z - mu * gz, mu * lam_b, mu * eps)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        beta = ((t - 1) / tn)[:, None]
        if restart:
            back = np.sum((z - xn) * (xn - x), 1) > 0
            beta[back] = 0.0; tn[back] = 1.0
        zn = xn + beta * (xn - x)
        Ax_prev, ATAx_prev, beta_prev = Ax_c, ATAx_c, beta
        x, z, t = xn, zn, tn
    k_star[~done] = max_iter; k_iter[~done] = max_iter; x_out[~done] = x[~done]
    return CertRun(k_star=k_star, x=x_out, time_s=time.perf_counter() - t0, violations=viol, checks=checks,
                   ratios=[np.concatenate(ratios)] if ratios else [], extra=dict(k_iter=k_iter, extra_ops=extra_ops))


def banach_guard(d_norm, e_norm, rho, n, dtype=np.float32):
    u = np.finfo(dtype).eps / 2
    return (rho * np.asarray(d_norm, np.float64) * (1 + (n + 3) * u) + np.asarray(e_norm, np.float64)) / (1 - rho)


def rounding_guard(x_new, x_old, rho, dtype=np.float32):
    u = np.finfo(dtype).eps / 2
    n = x_new.shape[-1]
    d = np.linalg.norm(x_new.astype(np.float64) - x_old.astype(np.float64), axis=-1)
    nx = np.linalg.norm(x_new.astype(np.float64), axis=-1) + np.linalg.norm(x_old.astype(np.float64), axis=-1)
    return (rho / (1 - rho)) * ((n + 2) * u * d + u * nx * np.sqrt(n))


def coordinate_descent(A, y, lam, eps, n_sweeps=20000, tol=1e-13):
    Ad = as_operator(A).to_dense(); y = np.asarray(y, np.float64)
    m, n = Ad.shape; x = np.zeros(n); r = y.copy(); col2 = np.sum(Ad * Ad, 0)
    for _ in range(n_sweeps):
        max_change = 0.0
        for j in range(n):
            aj = Ad[:, j]
            rho_j = aj @ r + col2[j] * x[j]
            new = np.sign(rho_j) * max(abs(rho_j) - lam, 0.0) / (col2[j] + eps)
            if new != x[j]:
                r -= aj * (new - x[j]); max_change = max(max_change, abs(new - x[j])); x[j] = new
        if max_change <= tol:
            break
    return x
