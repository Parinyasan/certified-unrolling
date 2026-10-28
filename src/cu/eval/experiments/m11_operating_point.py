from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certified_solve import fista_gap
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.models.lista import make_head
from cu.train.train_head import head_outputs, train_head

LAM_REL, EPS_REL = 0.01, 1e-3


def problem_cfg():
    cfg = copy.deepcopy(load_config("configs/synth_base.yaml")["problem"])
    cfg["lam"] = LAM_REL; cfg["eps_rel"] = EPS_REL
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="results/m11")
    args = ap.parse_args()
    cfg = problem_cfg()
    for seed in args.seeds:
        out = Path(args.out) / f"seed{seed}"; (out / "heads").mkdir(parents=True, exist_ok=True)
        A, Ytr, _, mtr = load_synthetic(cfg, args.n_train, seed=seed)
        Ad = A.to_dense(); smax2 = A.sigma_max() ** 2
        t0 = time.perf_counter()
        xh_tr, k_tr = fista_gap(Ad, Ytr, mtr["lam"], mtr["eps"], 1.0 / smax2, None, 1e-6)
        offline_s = time.perf_counter() - t0
        S = dict(seed=seed, lam_rel=LAM_REL, eps_rel=EPS_REL, n_train=int(Ytr.shape[0]), steps=args.steps,
                 offline_xhat_s=offline_s, offline_fista_pairs_mean=float(k_tr.mean()), heads={})
        print(f"seed {seed}: targets certified to 1e-6 in {offline_s:.0f}s ({k_tr.mean():.0f} pairs)", flush=True)
        for K in (8, 16):
            t0 = time.perf_counter()
            head = make_head("lista_cp", Ad, K, mu=1.0 / smax2)
            train_head(head, Ytr, xh_tr, mtr["lam"], dict(batch=64, lr=5e-4, steps_per_stage=args.steps, layerwise=True, loss="last"),
                       seed=seed, log=lambda s: None)
            torch.save(head.state_dict(), out / "heads" / f"cp_K{K}_xhat.pt")
            tr = time.perf_counter() - t0
            x0 = head_outputs(head, Ytr[:2000], mtr["lam"][:2000])
            d = np.linalg.norm(x0 - xh_tr[:2000], axis=1) / np.linalg.norm(xh_tr[:2000], axis=1)
            S["heads"][f"K{K}"] = dict(train_s=tr, delta_train_mean=float(d.mean()), delta_train_median=float(np.median(d)))
            print(f"seed {seed}: K={K} trained in {tr:.0f}s, train delta mean {d.mean():.3f}", flush=True)
        json.dump(S, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
