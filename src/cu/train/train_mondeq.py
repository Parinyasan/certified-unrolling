from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from cu.models.mondeq_tail import MonDEQTail, ZEncoder


def train_mondeq(tail: MonDEQTail, Y, X, cfg: dict, Y_val=None, X_val=None, log=print, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dtype = tail.P.dtype
    Yt, Xt = torch.as_tensor(Y, dtype=dtype), torch.as_tensor(X, dtype=dtype)
    N = Yt.shape[0]
    steps, bs = int(cfg.get("steps", 3000)), int(cfg.get("batch", 64))
    lr, L_max, lam_L = float(cfg.get("lr", 1e-3)), float(cfg.get("L_max", 1.5)), float(cfg.get("lam_L", 10.0))
    opt = torch.optim.Adam(tail.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(cfg.get("lr_decay_at", 0.7) * steps)], gamma=0.1)
    hist = dict(step=[], loss=[], L=[], rho_B=[], k_fp=[], val=[])
    t0 = time.perf_counter()
    run = None
    for it in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), generator=g)
        x, z, k = tail(Yt[idx], max_iter=int(cfg.get("max_iter", 100)), tol=float(cfg.get("tol", 1e-4)), return_k=True)
        L = tail.lipschitz(exact=False)
        loss = ((x - Xt[idx]) ** 2).sum(1).mean() + lam_L * F.relu(L - L_max) ** 2
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if it % max(1, steps // 20) == 0 or it == steps:
            with torch.no_grad():
                Le = float(tail.lipschitz(exact=True)); rb = tail.rho_B(Le)
                val = evaluate_mondeq(tail, Y_val, X_val) if Y_val is not None else float("nan")
            hist["step"].append(it); hist["loss"].append(run); hist["L"].append(Le); hist["rho_B"].append(rb); hist["k_fp"].append(k); hist["val"].append(val)
            log(f"  [mondeq {it:5d}/{steps}] loss(ema)={run:.4e} L={Le:.3f} rho_B={rb:.4f} fp-iters={k:3d} val-NMSE={val:7.2f} dB  t={time.perf_counter()-t0:.0f}s")
    hist["total_time_s"] = time.perf_counter() - t0
    return hist


@torch.no_grad()
def evaluate_mondeq(tail: MonDEQTail, Y, X, max_iter=500, tol=1e-7) -> float:
    dtype = tail.P.dtype
    x = tail(torch.as_tensor(Y, dtype=dtype), max_iter=max_iter, tol=tol)
    X = torch.as_tensor(X, dtype=dtype)
    return float(10 * torch.log10(((x - X) ** 2).sum(1).mean() / (X**2).sum(1).mean()))


def train_encoder(enc: ZEncoder, Y, Zstar, cfg: dict, log=print, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dtype = enc.lin.weight.dtype
    Yt, Zt = torch.as_tensor(Y, dtype=dtype), torch.as_tensor(Zstar, dtype=dtype)
    N = Yt.shape[0]
    steps, bs = int(cfg.get("steps", 2000)), int(cfg.get("batch", 64))
    opt = torch.optim.Adam(enc.parameters(), lr=float(cfg.get("lr", 1e-3)))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.7 * steps)], gamma=0.1)
    t0 = time.perf_counter(); run = None
    for it in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), generator=g)
        loss = ((enc(Yt[idx]) - Zt[idx]) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
    log(f"  [encoder] final loss(ema)={run:.4e}  t={time.perf_counter()-t0:.0f}s")
    return dict(loss=run, time_s=time.perf_counter() - t0)


@torch.no_grad()
def encoder_outputs(enc: ZEncoder, Y) -> np.ndarray:
    return enc(torch.as_tensor(Y, dtype=enc.lin.weight.dtype)).double().numpy()
