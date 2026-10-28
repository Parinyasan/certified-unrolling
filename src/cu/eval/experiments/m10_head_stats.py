from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cu.algos.certified_solve import head_cast, head_forward, head_params
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.eval.experiments.m5r_e6 import load
from cu.eval.experiments.m6_endtoend import REAL, load_problem
from cu.models.lista import make_head


def stats(H, Y, lam, xh):
    x0 = head_forward(H, Y, lam)
    nx = np.linalg.norm(xh, axis=1)
    d = np.linalg.norm(x0 - xh, axis=1)
    return dict(E_delta=float(np.mean(d / nx)), median_delta=float(np.median(d / nx)),
                cs_bound=float(np.sqrt(np.mean(d ** 2)) * np.sqrt(np.mean(nx ** -2.0))), cv_norm=float(nx.std() / nx.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "ecg_compressive", "ecg_denoise_overcomplete", "cwru_or007", "cwru_ir007"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out", default="results/m10/head_stats.json")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    S = json.load(open(out)) if out.exists() else {}
    for name in args.datasets:
        seeds = [0, 1, 2, 3, 4] if name == "synthetic" else ([0] if name == "ecg_denoise_overcomplete" else [0, 1, 2])
        for seed in seeds:
            A, Y, lam, eps, heads = load_problem(name, seed, 1000 if name == "synthetic" else 500)
            if name == "synthetic":
                cfg = load_config("configs/synth_base.yaml")
                _, Ytr, _, mtr = load_synthetic(cfg["problem"], 25600, seed=seed)
                heads = dict(heads, K16_xstar=("lista_cp", 16, Path("results/m3") / f"seed{seed}" / "heads" / "cp_K16_xstar.pt"))
            else:
                spec = REAL[name]
                _, Ytr, _, mtr = load(spec["kind"], spec["cfg"], 6400, seed, "train")
            Ytr, lamtr = Ytr[: args.n_train], np.asarray(mtr["lam"])[: args.n_train]
            tail = ContractiveTail(A, lam, eps)
            xh, _, _ = tail.solve_fixed_point(Y, tol=1e-10)
            xh_tr, _, _ = ContractiveTail(A, lamtr, eps).solve_fixed_point(Ytr, tol=1e-10)
            Ad = A.to_dense()
            ent = {}
            for key, (kind, K, path) in heads.items():
                if not Path(path).exists():
                    continue
                h = make_head(kind, Ad, K, mu=1.0 / A.sigma_max() ** 2)
                h.load_state_dict(torch.load(path))
                H = head_cast(head_params(h), np.float64)
                te = stats(H, Y, lam, xh); tr = stats(H, Ytr, lamtr, xh_tr)
                ent[key] = dict(test=te, train=tr, cs_ratio=te["cs_bound"] / te["E_delta"], n_train=int(Ytr.shape[0]), n_test=int(Y.shape[0]))
                print(f"{name} s{seed} {key:14s} E[delta] train {tr['E_delta']:.4f} test {te['E_delta']:.4f}  CS ratio {ent[key]['cs_ratio']:.3f}  CV {te['cv_norm']:.3f}", flush=True)
            S[f"{name}/seed{seed}"] = ent
            with open(out, "w") as f:
                json.dump(S, f, indent=1)


if __name__ == "__main__":
    main()
