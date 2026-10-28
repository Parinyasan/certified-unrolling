from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cu.algos.certified_solve import fista_gap, ista_banach
from cu.algos.ista import fista
from cu.config import load_config
from cu.data.synthetic import load_synthetic

LAMS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
EPSS = [1e-4, 1e-3, 1e-2]


def nmse_db(X, T):
    return float(10 * np.log10(np.sum((X - T) ** 2) / np.sum(T ** 2)))


def problem(name, seed, split, n):
    if name == "synthetic":
        cfg = load_config("configs/synth_base.yaml")
        A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=seed)
        off = {"test": 1000, "val": 2000}[split]
        _, Y, X, _ = load_synthetic(cfg["problem"], n, seed=seed + off, A=A)
        Ad = A.to_dense()
        return Ad, Y, (lambda Z: nmse_db(Z, X))
    from cu.data.ecg import load_ecg
    cfg = dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2)
    A, _, _, _ = load_ecg(cfg, 6400, seed, "train")
    off, spl = ({"test": (1000, "test"), "val": (3000, "train")})[split]
    _, Y, _, m = load_ecg(cfg, n, seed + off, spl)
    Psi, S = m["Psi"], m["S"]
    return A.to_dense(), Y, (lambda Z: nmse_db(Z @ Psi.T, S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "ecg_compressive"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: 0-4 synthetic, 0-2 ECG")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--out", default="results/m8")
    ap.add_argument("--force", action="store_true", help="recompute seeds whose output exists")
    args = ap.parse_args()
    for name in args.datasets:
        seeds = args.seeds or ([0, 1, 2, 3, 4] if name == "synthetic" else [0, 1, 2])
        for seed in seeds:
            T0 = time.perf_counter()
            out = Path(args.out) / name; out.mkdir(parents=True, exist_ok=True)
            if (out / f"seed{seed}.json").exists() and not args.force:
                print(f"skip {name} seed {seed} (exists)", flush=True); continue
            S = dict(dataset=name, seed=seed, lams=LAMS, epss=EPSS, grid={})
            splits = {"val": problem(name, seed, "val", args.n_val),
                      "test": problem(name, seed, "test", args.n_test or (1000 if name == "synthetic" else 500))}
            Ad = splits["test"][0]; smax2 = float(np.linalg.norm(Ad, 2) ** 2)
            for lr in LAMS:
                for spl, (_, Y, met) in splits.items():
                    lam = lr * np.max(np.abs(Y @ Ad), axis=1)
                    xl = fista(Ad, Y, lam, 3000, mu=1.0 / smax2)
                    S["grid"].setdefault(f"{lr}", {}).setdefault(spl, {})["lasso_db"] = met(xl)
                    for er in EPSS:
                        eps = er * smax2; mu = 2.0 / smax2; rho = 1.0 / (1.0 + mu * eps)
                        xo, _ = fista_gap(Ad, Y, lam, eps, 1.0 / smax2, None, 1e-4)
                        ent = dict(certified_db=met(xo))
                        if spl == "test":
                            _, ki = ista_banach(Ad, Y, lam, eps, mu, rho, None, 1e-3)
                            _, kf = fista_gap(Ad, Y, lam, eps, 1.0 / smax2, None, 1e-3)
                            ent.update(k_ista=float(ki.mean()), k_fista=float(kf.mean()), rho=rho)
                        S["grid"][f"{lr}"][spl][f"{er}"] = ent
                g = S["grid"][f"{lr}"]["test"]
                print(f"{name} s{seed} lam {lr:<6g} LASSO {g['lasso_db']:7.2f} dB | " + " | ".join(
                    f"eps {er:g}: {g[f'{er}']['certified_db']:7.2f} dB, {g[f'{er}']['k_fista']:6.1f}/{g[f'{er}']['k_ista']:6.1f} pairs" for er in EPSS), flush=True)
            S["selected"] = {}
            for er in EPSS:
                best = max(LAMS, key=lambda lr: -S["grid"][f"{lr}"]["val"][f"{er}"]["certified_db"])
                S["selected"][f"{er}"] = dict(lam_rel=best, **S["grid"][f"{best}"]["test"][f"{er}"])
            bestl = max(LAMS, key=lambda lr: -S["grid"][f"{lr}"]["val"]["lasso_db"])
            S["selected"]["lasso"] = dict(lam_rel=bestl, certified_db=S["grid"][f"{bestl}"]["test"]["lasso_db"])
            S["time_s"] = time.perf_counter() - T0
            with open(out / f"seed{seed}.json", "w") as f:
                json.dump(S, f, indent=1)
            print(f"selected: {S['selected']}  ({S['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
