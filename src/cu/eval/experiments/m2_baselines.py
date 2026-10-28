from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cu.algos.ista import fista, ista
from cu.config import config_hash, load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db
from cu.models.lista import make_head
from cu.train.train_head import evaluate, head_outputs, train_head


def nmse_trace(A, Y, X, lam, n_iter, algo, eps=0.0):
    _, tr = algo(A, Y, lam, n_iter, eps=eps, return_trace=True)
    return [nmse_db(x, X) for x in tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--heads", nargs="+", default=["lista_cp", "lista_cpss", "alista", "tied_lista_cp"])
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="Adam steps per stage (override)")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-iter-trace", type=int, default=300)
    ap.add_argument("--out", default="results/m2")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    cfg = load_config(args.config)
    tr_cfg = dict(cfg["head"]["train"])
    if args.n_train:
        tr_cfg["n_train"] = args.n_train
    if args.steps:
        tr_cfg["steps_per_stage"] = args.steps
    tr_cfg.setdefault("steps_per_stage", int(tr_cfg["n_train"] // tr_cfg["batch"] * tr_cfg.get("epochs_per_layer", 1) / 3))
    h = config_hash(dict(problem=cfg["problem"], train=tr_cfg, K=args.K, heads=args.heads, seed=args.seed, n_test=args.n_test))
    out = Path(args.out) / h
    (out / "heads").mkdir(parents=True, exist_ok=True)
    print(f"=== M2 baselines  hash={h}  seed={args.seed}  K={args.K}  train={tr_cfg}")
    t_all = time.perf_counter()

    A, Ytr, Xtr, mtr = load_synthetic(cfg["problem"], int(tr_cfg["n_train"]), seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    _, Yva, Xva, mva = load_synthetic(cfg["problem"], 500, seed=args.seed + 2000, A=A)
    Ad = A.to_dense()
    mu = 1.0 / A.sigma_max() ** 2
    print(f"A {A.shape} smax={A.sigma_max():.4f}; train {Ytr.shape[0]}, test {Yte.shape[0]}")

    t0 = time.perf_counter()
    traces = dict(
        ISTA=nmse_trace(A, Yte, Xte, mte["lam"], args.n_iter_trace, ista),
        FISTA=nmse_trace(A, Yte, Xte, mte["lam"], args.n_iter_trace, fista),
        ENISTA=nmse_trace(A, Yte, Xte, mte["lam"], args.n_iter_trace, ista, eps=mte["eps"]),
    )
    print(f"classical traces ({args.n_iter_trace} it): " + ", ".join(f"{k}@16={v[16]:.2f} dB, @{args.n_iter_trace}={v[-1]:.2f} dB" for k, v in traces.items())
          + f"  [{time.perf_counter()-t0:.1f}s]")

    summary = dict(hash=h, seed=args.seed, K=args.K, train=tr_cfg, m=A.m, n=A.n, traces=traces, heads={})
    per_layer = {}
    for kind in args.heads:
        t0 = time.perf_counter()
        head = make_head(kind, Ad, args.K, mu=mu)
        print(f"--- training {kind} (K={args.K}, params={sum(p.numel() for p in head.parameters())})")
        hist = train_head(head, Ytr, Xtr, mtr["lam"], tr_cfg, Y_val=Yva, T_val=Xva, lam_val=mva["lam"], seed=args.seed)
        nm = [evaluate(head, Yte, Xte, mte["lam"], n_layers=k + 1) for k in range(args.K)]
        per_layer[kind] = nm
        dt = time.perf_counter() - t0
        print(f"{kind}: test NMSE per layer = {np.round(nm, 2).tolist()}  final {nm[-1]:.2f} dB  [{dt:.1f}s]")
        torch.save(head.state_dict(), out / "heads" / f"{kind}_K{args.K}.pt")
        summary["heads"][kind] = dict(nmse_per_layer=nm, final_nmse_db=nm[-1], train_time_s=dt, hist=hist)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    it = np.arange(args.n_iter_trace + 1)
    for k, v in traces.items():
        ax.plot(it, v, lw=1.2, label=f"{k} (iterations)")
    for kind, nm in per_layer.items():
        ax.plot(np.arange(1, args.K + 1), nm, "o-", ms=3, lw=1.2, label=f"{kind} (layers)")
    ax.set_xscale("log"); ax.set_xlim(1, args.n_iter_trace)
    ax.set_xlabel("iterations / layers"); ax.set_ylabel(r"NMSE vs $x^\star$ (dB)")
    ax.set_title(f"synthetic {A.m}x{A.n}, SNR {cfg['problem']['snr_db']} dB, seed {args.seed}")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "nmse_vs_depth.png", dpi=150)
    summary["wallclock_s"] = time.perf_counter() - t_all
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"figure {out/'nmse_vs_depth.png'}; TOTAL wall-clock {summary['wallclock_s']:.1f}s")


if __name__ == "__main__":
    main()
