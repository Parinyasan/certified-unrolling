from __future__ import annotations

import time

import numba
import numpy as np

GAP_FLOOR_C = 16.0
_U64 = float(np.finfo(np.float64).eps)


@numba.njit(cache=True, fastmath=False)
def _soft_rows(v, th, add, out):
    B, n = v.shape
    use_add = add.shape[0] == B
    for i in range(B):
        t = th[i]
        for j in range(n):
            a = v[i, j] + add[i, j] if use_add else v[i, j]
            if a > t:
                out[i, j] = a - t
            elif a < -t:
                out[i, j] = a + t
            else:
                out[i, j] = 0.0


@numba.njit(cache=True, fastmath=False)
def _ista_kernel(x, G, mu, th, c, tau, fac, xn, C, fired):
    B, n = x.shape
    inv = 1.0 / (1.0 + c)
    for i in range(B):
        t = th[i]; d2 = 0.0; n2 = 0.0
        for j in range(n):
            v = x[i, j] - mu * G[i, j]
            if v > t:
                w = (v - t) * inv
            elif v < -t:
                w = (v + t) * inv
            else:
                w = 0.0
            xn[i, j] = w
            dd = w - x[i, j]
            d2 += dd * dd; n2 += w * w
        Ci = fac * np.sqrt(d2)
        C[i] = Ci
        fired[i] = Ci <= tau * (np.sqrt(n2) - Ci)


@numba.njit(cache=True, fastmath=False)
def _gap_kernel(x, Az, gz, Axp, ATAxp, bp, y, ATy, lam, eps, refresh, Axr, ATAxr, Axc, ATAxc, tau, floor_c, C, fired):
    B, n = x.shape
    m = y.shape[1]
    for i in range(B):
        if refresh:
            for k in range(m):
                Axc[i, k] = Axr[i, k]
            for j in range(n):
                ATAxc[i, j] = ATAxr[i, j]
        else:
            b = bp[i]; w = 1.0 / (1.0 + b)
            for k in range(m):
                Axc[i, k] = w * (Az[i, k] + b * Axp[i, k])
            for j in range(n):
                ATAxc[i, j] = w * (gz[i, j] + ATy[i, j] + b * ATAxp[i, j])
        rr = 0.0; ry = 0.0
        for k in range(m):
            r = Axc[i, k] - y[i, k]
            rr += r * r; ry += r * y[i, k]
        l1 = 0.0; xx = 0.0; ee = 0.0; la = lam[i]
        for j in range(n):
            xi = x[i, j]
            l1 += abs(xi); xx += xi * xi
            a = abs(ATAxc[i, j] - ATy[i, j]) - la
            if a > 0.0:
                ee += a * a
        P = 0.5 * rr + la * l1 + 0.5 * eps * xx
        D = -ry - 0.5 * rr - ee / (2.0 * eps)
        g = P - D
        fl = floor_c * 2.220446049250313e-16 * (abs(P) + abs(D))
        if g < fl:
            g = fl
        Ci = np.sqrt(2.0 * g / eps)
        C[i] = Ci
        fired[i] = Ci <= tau * (np.sqrt(xx) - Ci)


@numba.njit(cache=True, fastmath=False)
def _fista_step_kernel(z, gz, x, t, lam, mu, eps, restart, xn, zn, tn, beta):
    B, n = z.shape
    inv = 1.0 / (1.0 + mu * eps)
    for i in range(B):
        th = mu * lam[i]
        dot = 0.0
        for j in range(n):
            v = z[i, j] - mu * gz[i, j]
            if v > th:
                w = (v - th) * inv
            elif v < -th:
                w = (v + th) * inv
            else:
                w = 0.0
            xn[i, j] = w
            dot += (z[i, j] - w) * (w - x[i, j])
        tni = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t[i] * t[i]))
        b = (t[i] - 1.0) / tni
        if restart and dot > 0.0:
            b = 0.0; tni = 1.0
        tn[i] = tni; beta[i] = b
        for j in range(n):
            zn[i, j] = xn[i, j] + b * (xn[i, j] - x[i, j])


def head_params(head) -> dict:
    import torch.nn.functional as F

    kind = type(head).__name__
    p = dict(kind=kind, K=head.K, A=head.A.detach().cpu().numpy().astype(np.float64),
             t=F.softplus(head.t_raw.detach()).cpu().numpy().astype(np.float64))
    if kind == "LISTACP":
        p["W"] = [w.detach().cpu().numpy().astype(np.float64) for w in head.W]
    elif kind == "TiedLISTACP":
        p["W"] = [head.W.detach().cpu().numpy().astype(np.float64)] * head.K
    elif kind == "ConvLISTA":
        p["W"] = [head.dense_W(k) for k in range(head.K)]
    elif kind == "ALISTA":
        p["Wa"] = head.W.detach().cpu().numpy().astype(np.float64)
        p["gamma"] = F.softplus(head.gamma_raw.detach()).cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"head type {kind} is not supported by the timing path")
    return p


def _ftz(a: np.ndarray) -> np.ndarray:
    fi = np.finfo(a.dtype)
    a[np.abs(a) < max(fi.tiny, float(np.abs(a).max()) * (fi.eps / 2) ** 2)] = 0
    return a


def head_cast(p: dict, dtype) -> dict:
    q = dict(p)
    q["A"] = _ftz(np.array(p["A"], dtype=dtype, order="C"))
    q["AT"] = _ftz(np.array(p["A"].T, dtype=dtype, order="C"))
    if "W" in p:
        q["W"] = [_ftz(np.array(w, dtype=dtype, order="C")) for w in p["W"]]
    if "Wa" in p:
        q["Wa"] = _ftz(np.array(p["Wa"], dtype=dtype, order="C"))
    q["dtype"] = dtype
    return q


def head_forward(p: dict, y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    dt = p.get("dtype", np.float64)
    y = np.ascontiguousarray(y, dtype=dt); lam = np.asarray(lam, dt).reshape(-1)
    B = y.shape[0]; n = p["A"].shape[1]
    x = np.zeros((B, n), dt); empty = np.zeros((0, 0), dt)
    for k in range(p["K"]):
        r = y if k == 0 else y - x @ p["AT"]
        g = (r @ p["W"][k]) if "W" in p else (dt(p["gamma"][k]) * (r @ p["Wa"]))
        _soft_rows(g, (p["t"][k] * lam).astype(dt), empty if k == 0 else x, x)
    return x


def ista_banach(A: np.ndarray, y: np.ndarray, lam: np.ndarray, eps: float, mu: float, rho: float, x0, tau: float,
                max_iter: int = 100_000):
    A = np.ascontiguousarray(A, np.float64); AT = np.ascontiguousarray(A.T)
    y = np.ascontiguousarray(y, np.float64)
    B, n = y.shape[0], A.shape[1]; m = A.shape[0]
    X = np.zeros((B, n)) if x0 is None else np.array(x0, np.float64, order="C", copy=True)
    XN = np.empty((B, n)); R = np.empty((B, m)); G = np.empty((B, n)); Y = y.copy()
    TH = mu * np.asarray(lam, np.float64).reshape(-1).copy(); c = mu * eps; fac = rho / (1 - rho)
    C = np.empty(B); FI = np.empty(B, np.bool_)
    out = np.empty((B, n)); kst = np.full(B, max_iter); act = np.arange(B); nb = B
    for k in range(1, max_iter + 1):
        x, xn, r, g = X[:nb], XN[:nb], R[:nb], G[:nb]
        np.matmul(x, AT, out=r); np.subtract(r, Y[:nb], out=r); np.matmul(r, A, out=g)
        _ista_kernel(x, g, mu, TH[:nb], c, tau, fac, xn, C[:nb], FI[:nb])
        fired = FI[:nb]
        if fired.any():
            out[act[fired]] = xn[fired]; kst[act[fired]] = k
            keep = ~fired; nk = int(keep.sum())
            act = act[keep]
            if nk == 0:
                break
            X[:nk] = xn[keep]; Y[:nk] = Y[:nb][keep]; TH[:nk] = TH[:nb][keep]
            nb = nk
        else:
            X, XN = XN, X
    if act.size:
        out[act] = X[:nb]
    return out, kst


def fista_gap(A: np.ndarray, y: np.ndarray, lam: np.ndarray, eps: float, mu: float, x0, tau: float,
              max_iter: int = 100_000, refresh_every: int = 50, restart: bool = True):
    A = np.ascontiguousarray(A, np.float64); AT = np.ascontiguousarray(A.T)
    y = np.ascontiguousarray(y, np.float64)
    B, n = y.shape[0], A.shape[1]; m = A.shape[0]
    X = np.zeros((B, n)) if x0 is None else np.array(x0, np.float64, order="C", copy=True)
    Z = X.copy(); XN = np.empty((B, n)); ZN = np.empty((B, n))
    T = np.ones(B); TN = np.empty(B); BETA = np.empty(B); BP = np.zeros(B)
    Y = y.copy(); LA = np.asarray(lam, np.float64).reshape(-1).copy(); ATY = y @ A
    AZ = np.empty((B, m)); RZ = np.empty((B, m)); GZ = np.empty((B, n))
    AXP = np.zeros((B, m)); ATAXP = np.zeros((B, n)); AXC = np.empty((B, m)); ATAXC = np.empty((B, n))
    AXR = np.empty((B, m)); ATAXR = np.empty((B, n)); empty = np.zeros((0, 0))
    C = np.empty(B); FI = np.empty(B, np.bool_)
    out = np.empty((B, n)); kst = np.full(B, max_iter); act = np.arange(B); nb = B
    for k in range(1, max_iter + 1):
        x, z, az, rz, gz = X[:nb], Z[:nb], AZ[:nb], RZ[:nb], GZ[:nb]
        np.matmul(z, AT, out=az); np.subtract(az, Y[:nb], out=rz); np.matmul(rz, A, out=gz)
        refresh = bool(refresh_every) and k % refresh_every == 0
        if refresh:
            axr, ataxr = AXR[:nb], ATAXR[:nb]
            np.matmul(x, AT, out=axr); np.matmul(axr, A, out=ataxr)
        else:
            axr, ataxr = empty, empty
        _gap_kernel(x, az, gz, AXP[:nb], ATAXP[:nb], BP[:nb], Y[:nb], ATY[:nb], LA[:nb], eps, refresh, axr, ataxr,
                    AXC[:nb], ATAXC[:nb], tau, GAP_FLOOR_C, C[:nb], FI[:nb])
        fired = FI[:nb]
        if fired.any():
            out[act[fired]] = x[fired]
            kst[act[fired]] = k + 1 + (k // refresh_every if refresh_every else 0)
            keep = ~fired; nk = int(keep.sum())
            act = act[keep]
            if nk == 0:
                break
            for buf in (X, Z, GZ, AXC, ATAXC, Y, ATY):
                buf[:nk] = buf[:nb][keep]
            T[:nk] = T[:nb][keep]; LA[:nk] = LA[:nb][keep]
            nb = nk
        _fista_step_kernel(Z[:nb], GZ[:nb], X[:nb], T[:nb], LA[:nb], mu, eps, restart, XN[:nb], ZN[:nb], TN[:nb], BETA[:nb])
        AXP, AXC = AXC, AXP; ATAXP, ATAXC = ATAXC, ATAXP; BP, BETA = BETA, BP
        X, XN = XN, X; Z, ZN = ZN, Z; T, TN = TN, T
    if act.size:
        out[act] = X[:nb]
    return out, kst


@numba.njit(cache=True, fastmath=False)
def ista_banach_1(A, AT, y, lam, eps, mu, rho, x0, tau, max_iter):
    m, n = A.shape
    x = x0.copy(); xn = np.empty(n); r = np.empty(m); g = np.empty(n)
    th = mu * lam; inv = 1.0 / (1.0 + mu * eps); fac = rho / (1.0 - rho)
    for k in range(1, max_iter + 1):
        np.dot(A, x, r)
        for i in range(m):
            r[i] -= y[i]
        np.dot(AT, r, g)
        d2 = 0.0; n2 = 0.0
        for j in range(n):
            v = x[j] - mu * g[j]
            if v > th:
                w = (v - th) * inv
            elif v < -th:
                w = (v + th) * inv
            else:
                w = 0.0
            xn[j] = w
            dd = w - x[j]
            d2 += dd * dd; n2 += w * w
        C = fac * np.sqrt(d2)
        tmp = x; x = xn; xn = tmp
        if C <= tau * (np.sqrt(n2) - C):
            return x, k
    return x, max_iter


@numba.njit(cache=True, fastmath=False)
def fista_gap_1(A, AT, y, ATy, lam, eps, mu, x0, tau, max_iter, refresh_every):
    m, n = A.shape
    x = x0.copy(); z = x0.copy(); xn = np.empty(n); zn = np.empty(n)
    az = np.empty(m); rz = np.empty(m); gz = np.empty(n)
    axp = np.zeros(m); ataxp = np.zeros(n); axc = np.empty(m); ataxc = np.empty(n)
    t = 1.0; bp = 0.0; inv = 1.0 / (1.0 + mu * eps); th = mu * lam
    for k in range(1, max_iter + 1):
        np.dot(A, z, az)
        for i in range(m):
            rz[i] = az[i] - y[i]
        np.dot(AT, rz, gz)
        if refresh_every > 0 and k % refresh_every == 0:
            np.dot(A, x, axc); np.dot(AT, axc, ataxc)
        else:
            w = 1.0 / (1.0 + bp)
            for i in range(m):
                axc[i] = w * (az[i] + bp * axp[i])
            for j in range(n):
                ataxc[j] = w * (gz[j] + ATy[j] + bp * ataxp[j])
        rr = 0.0; ry = 0.0
        for i in range(m):
            rv = axc[i] - y[i]
            rr += rv * rv; ry += rv * y[i]
        l1 = 0.0; xx = 0.0; ee = 0.0
        for j in range(n):
            xj = x[j]
            l1 += abs(xj); xx += xj * xj
            a = abs(ataxc[j] - ATy[j]) - lam
            if a > 0.0:
                ee += a * a
        P = 0.5 * rr + lam * l1 + 0.5 * eps * xx
        D = -ry - 0.5 * rr - ee / (2.0 * eps)
        gap = P - D
        fl = 16.0 * 2.220446049250313e-16 * (abs(P) + abs(D))
        if gap < fl:
            gap = fl
        C = np.sqrt(2.0 * gap / eps)
        if C <= tau * (np.sqrt(xx) - C):
            return x, k + 1 + (k // refresh_every if refresh_every > 0 else 0)
        dot = 0.0
        for j in range(n):
            v = z[j] - mu * gz[j]
            if v > th:
                w2 = (v - th) * inv
            elif v < -th:
                w2 = (v + th) * inv
            else:
                w2 = 0.0
            xn[j] = w2
            dot += (z[j] - w2) * (w2 - x[j])
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        b = (t - 1.0) / tn
        if dot > 0.0:
            b = 0.0; tn = 1.0
        for j in range(n):
            zn[j] = xn[j] + b * (xn[j] - x[j])
        tmp = axp; axp = axc; axc = tmp
        tmp2 = ataxp; ataxp = ataxc; ataxc = tmp2
        bp = b; t = tn
        tmp3 = x; x = xn; xn = tmp3
        tmp4 = z; z = zn; zn = tmp4
    return x, max_iter


@numba.njit(cache=True, fastmath=False)
def head_forward_1(AT, Ws, coef, t, lam, y):
    K = Ws.shape[0]; m = Ws.shape[1]; n = Ws.shape[2]
    x = np.zeros(n, y.dtype); r = np.empty(m, y.dtype); g = np.empty(n, y.dtype)
    for k in range(K):
        if k == 0:
            for i in range(m):
                r[i] = y[i]
        else:
            np.dot(x, AT, r)
            for i in range(m):
                r[i] = y[i] - r[i]
        np.dot(r, Ws[k], g)
        th = t[k] * lam; ck = coef[k]
        for j in range(n):
            v = x[j] + ck * g[j]
            if v > th:
                x[j] = v - th
            elif v < -th:
                x[j] = v + th
            else:
                x[j] = 0.0
    return x


def head_stack(p: dict):
    dt = p.get("dtype", np.float64)
    if "W" in p:
        Ws = np.ascontiguousarray(np.stack(p["W"]), dtype=dt); coef = np.ones(p["K"], dt)
    else:
        Ws = np.ascontiguousarray(np.stack([p["Wa"]] * p["K"]), dtype=dt); coef = np.asarray(p["gamma"], dt)
    return np.ascontiguousarray(p["AT"], dt), Ws, coef, np.asarray(p["t"], dt)


def timed(fn, *args, repeats: int = 3, **kw):
    res = fn(*args, **kw)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); res = fn(*args, **kw); ts.append(time.perf_counter() - t0)
    return res, float(np.median(ts))
