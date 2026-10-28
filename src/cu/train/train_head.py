from __future__ import annotations

import time

import numpy as np
import torch

from cu.models.lista import UnrolledHead


def _loss(xs, target, mode):
    if mode == "last":
        return ((xs[-1] - target) ** 2).sum(1).mean()
    if mode == "all_layers":
        return sum(((x - target) ** 2).sum(1).mean() for x in xs)
    raise ValueError(mode)


def _params_upto(head: UnrolledHead, k: int):
    seen, out = set(), []
    for j in range(k + 1):
        for p in head.layer_params(j):
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def train_head(head: UnrolledHead, Y: np.ndarray, T: np.ndarray, lam: np.ndarray, cfg: dict, log=print,
               Y_val=None, T_val=None, lam_val=None, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    dtype = head.A.dtype
    Yt, Tt = torch.as_tensor(Y, dtype=dtype), torch.as_tensor(T, dtype=dtype)
    lt = torch.as_tensor(np.asarray(lam), dtype=dtype)
    N = Yt.shape[0]
    bs = int(cfg.get("batch", 64))
    lr0 = float(cfg.get("lr", 5e-4))
    steps = int(cfg.get("steps_per_stage", 400))
    mode = cfg.get("loss", "last")
    layerwise = bool(cfg.get("layerwise", True))
    clip = cfg.get("grad_clip", None)
    g = torch.Generator().manual_seed(seed)
    hist = dict(stage=[], loss=[], val=[], time_s=[])
    t0 = time.perf_counter()

    def run_stage(params, lr, n_layers, tag):
        opt = torch.optim.Adam(params, lr=lr)
        head.train()
        run = 0.0
        for it in range(steps):
            idx = torch.randint(0, N, (bs,), generator=g)
            xs = head(Yt[idx], lt[idx], n_layers=n_layers, return_all=True)
            loss = _loss(xs, Tt[idx], mode)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        val = evaluate(head, Y_val, T_val, lam_val, n_layers) if Y_val is not None else float("nan")
        hist["stage"].append(tag); hist["loss"].append(run); hist["val"].append(val); hist["time_s"].append(time.perf_counter() - t0)
        log(f"  [{tag:>14s}] lr={lr:.1e} layers={n_layers:2d} train-loss(ema)={run:.4e} val-NMSE={val:7.2f} dB  t={time.perf_counter()-t0:6.1f}s")

    K = head.K
    if layerwise:
        for k in range(K):
            run_stage(head.layer_params(k), lr0, k + 1, f"L{k} only")
            run_stage(_params_upto(head, k), 0.2 * lr0, k + 1, f"L0-{k} 0.2lr")
            run_stage(_params_upto(head, k), 0.02 * lr0, k + 1, f"L0-{k} 0.02lr")
    else:
        for f in (1.0, 0.2, 0.02):
            for _ in range(K):
                run_stage(list(head.parameters()), f * lr0, K, f"all {f:g}lr")
    head.eval()
    hist["total_time_s"] = time.perf_counter() - t0
    return hist


@torch.no_grad()
def evaluate(head: UnrolledHead, Y, T, lam, n_layers=None) -> float:
    head.eval()
    dtype = head.A.dtype
    x = head(torch.as_tensor(Y, dtype=dtype), torch.as_tensor(np.asarray(lam), dtype=dtype), n_layers=n_layers)
    T = torch.as_tensor(T, dtype=dtype)
    return float(10 * torch.log10(((x - T) ** 2).sum(1).mean() / (T**2).sum(1).mean()))


@torch.no_grad()
def head_outputs(head: UnrolledHead, Y, lam, n_layers=None) -> np.ndarray:
    head.eval()
    dtype = head.A.dtype
    return head(torch.as_tensor(Y, dtype=dtype), torch.as_tensor(np.asarray(lam), dtype=dtype), n_layers=n_layers).double().numpy()
