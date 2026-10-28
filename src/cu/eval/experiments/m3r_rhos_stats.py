from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cu.algos.certificates import rho_support, run_ista_certified
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic

TAUS = [1e-2, 1e-3, 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-oracle", type=int, default=200, help="instances for the oracle rho_S on supp(xhat)")
    ap.add_argument("--out", default="results/m3r")
    args = ap.parse_args()
    out = Path(args.out) / f"seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; lam = mte["lam"]
    tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
    xhat, _ = compute_and_cache_xhat(Yte, A, lam, eps, mu=mu, tol=1e-12, verbose=False)
    supp = (xhat != 0).sum(1)
    t0 = time.perf_counter()
    oracle = np.array([rho_support(Ad, xhat[i] != 0, mu, eps) for i in range(min(args.n_oracle, Yte.shape[0]))])
    res = dict(seed=args.seed, rho=rho, supp_size_median=float(np.median(supp)), rho_S_oracle_median=float(np.median(oracle)),
               rho_S_oracle_p90=float(np.percentile(oracle, 90)), taus={})
    for tau in TAUS:
        r = run_ista_certified(A, Yte, lam, eps, mu, None, tau, xhat=xhat, mode="rhoS", rho=rho)
        rs = r.extra["rhoS_at_stop"]; Ssz = r.extra["S_size_at_stop"]
        ok = np.isfinite(rs) & (Ssz > 0)
        res["taus"][str(tau)] = dict(
            k_star_mean=float(r.k_star.mean()),
            rhoS_at_stop_median=float(np.median(rs[ok])), rhoS_at_stop_p90=float(np.percentile(rs[ok], 90)),
            frac_global_fallback=float(np.mean(rs[ok] >= rho - 1e-12)),
            S_size_median=float(np.median(Ssz[ok])), S_over_supp_median=float(np.median(Ssz[ok] / np.maximum(supp[ok], 1))),
            cert_factor_median=float(np.median(rs[ok] / (1 - rs[ok]))))
        e = res["taus"][str(tau)]
        print(f"tau {tau:g}: k* {e['k_star_mean']:.1f}  rho_S(stop) median {e['rhoS_at_stop_median']:.3f} (oracle {res['rho_S_oracle_median']:.3f}, "
              f"global {rho:.3f})  |S| {e['S_size_median']:.0f} = {e['S_over_supp_median']:.2f} x |supp|  fallback {100*e['frac_global_fallback']:.0f}%  [{time.perf_counter()-t0:.0f}s]")
    res["time_s"] = time.perf_counter() - t0
    json.dump(res, open(out / "rhoS_stats.json", "w"), indent=1)
    print("wrote", out / "rhoS_stats.json")


if __name__ == "__main__":
    main()
