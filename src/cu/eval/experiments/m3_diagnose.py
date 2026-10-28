from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db
from cu.models.lista import make_head
from cu.train.train_head import head_outputs, train_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--eps-tols", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--variants", nargs="+", default=["cp_K16_s400", "cp_K16_s1600", "cpss_K16_s400", "cpss_K16_s1600", "cpss_K8_s400"])
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--out", default="results/m3/diag")
    args = ap.parse_args()
    out = Path(args.out); (out / "heads").mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config); T0 = time.perf_counter()

    A, Ytr, Xtr, mtr = load_synthetic(cfg["problem"], args.n_train, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    _, Yva, Xva, mva = load_synthetic(cfg["problem"], 500, seed=args.seed + 2000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; mu = 1.0 / A.sigma_max() ** 2
    tail = ContractiveTail(A, mte["lam"], eps, mu); rho = tail.rho()
    xhat_te, _ = compute_and_cache_xhat(Yte, A, mte["lam"], eps, mu=mu, tol=1e-12)
    xhat_tr, _ = compute_and_cache_xhat(Ytr, A, mtr["lam"], eps, mu=mu, tol=1e-10)
    xhat_va, _ = compute_and_cache_xhat(Yva, A, mva["lam"], eps, mu=mu, tol=1e-10)
    xn = np.linalg.norm(xhat_te, axis=1)
    S = dict(seed=args.seed, rho=rho, eps_tols=args.eps_tols, variants={}, ideal={}, cold={})

    def kstars(x0):
        out = {}
        for t in args.eps_tols:
            res = tail.run_until(x0, Yte, np.maximum(t * xn, 1e-7 * np.linalg.norm(Yte, axis=1)))
            out[str(t)] = float(res.k_star.mean())
        return out

    S["cold"] = kstars(None)
    print(f"rho={rho:.5f}; cold k*: {S['cold']}")
    rng = np.random.default_rng(1)
    for rel in [1e-1, 1e-2, 1e-3]:
        delta = rng.standard_normal(xhat_te.shape); delta *= rel * xn[:, None] / np.linalg.norm(delta, axis=1, keepdims=True)
        S["ideal"][str(rel)] = kstars(xhat_te + delta)
        d2 = delta * (xhat_te != 0); d2 *= rel * xn[:, None] / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-300)
        S["ideal"][str(rel) + "_on_support"] = kstars(xhat_te + d2)
        print(f"ideal x0 = xhat + {rel:g}||xhat|| (random / on-support): {S['ideal'][str(rel)]} / {S['ideal'][str(rel)+'_on_support']}")

    for v in args.variants:
        kind, Ks, ss = v.split("_"); K = int(Ks[1:]); steps = int(ss[1:])
        kind = "lista_cp" if kind == "cp" else "lista_cpss"
        t0 = time.perf_counter()
        head = make_head(kind, Ad, K, mu=mu)
        reuse = Path(args.m3_root) / f"seed{args.seed}" / "heads" / f"cp_K{K}_xhat.pt"
        if kind == "lista_cp" and steps == 400 and reuse.exists():
            head.load_state_dict(torch.load(reuse)); src = "reused M3 head"
        else:
            train_head(head, Ytr, xhat_tr, mtr["lam"], dict(batch=64, lr=5e-4, steps_per_stage=steps, layerwise=True, loss="last"),
                       Y_val=Yva, T_val=xhat_va, lam_val=mva["lam"], seed=args.seed, log=lambda s: None)
            torch.save(head.state_dict(), out / "heads" / f"{v}.pt"); src = "trained"
        x0 = head_outputs(head, Yte, mte["lam"])
        d = np.linalg.norm(x0 - xhat_te, axis=1) / xn
        ent = dict(kind=kind, K=K, steps_per_stage=steps, source=src, train_time_s=time.perf_counter() - t0,
                   nmse_vs_xhat_db=nmse_db(x0, xhat_te), nmse_vs_xstar_db=nmse_db(x0, Xte), d_rel_median=float(np.median(d)),
                   d_rel_p90=float(np.percentile(d, 90)), k_star=kstars(x0))
        ent["speedup"] = {t: S["cold"][t] / ent["k_star"][t] for t in ent["k_star"]}
        S["variants"][v] = ent
        print(f"{v:14s} ({src}): NMSE vs xhat {ent['nmse_vs_xhat_db']:6.2f} dB, d_rel median {ent['d_rel_median']:.4f}, k* {ent['k_star']}, "
              f"speedup {[round(s, 2) for s in ent['speedup'].values()]}  [{ent['train_time_s']:.0f}s]")
    S["timing_s"] = time.perf_counter() - T0
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(f"diagnosis done in {S['timing_s']:.0f}s")


if __name__ == "__main__":
    main()
