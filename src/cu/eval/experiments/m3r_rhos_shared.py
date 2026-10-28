from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certificates import rho_support, run_ista_certified
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db, percentiles
from cu.models.lista import make_head
from cu.train.train_head import head_outputs

TAUS = [1e-2, 1e-3, 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-oracle", type=int, default=200)
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--out", default="results/m3r")
    args = ap.parse_args()
    out = Path(args.out) / f"seed{args.seed}"
    for f in ("summary.json", "traces.npz", "rhoS_stats.json"):
        b = out / f.replace(".", "_pre_shared.")
        if not b.exists():
            shutil.copy(out / f, b)
    S = json.load(open(out / "summary.json")); traces = dict(np.load(out / "traces.npz"))
    cfg = load_config(args.config)
    A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; lam = mte["lam"]; smax2 = A.sigma_max() ** 2
    tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
    xhat, _ = compute_and_cache_xhat(Yte, A, lam, eps, mu=mu, tol=1e-12, verbose=False)
    xn = np.linalg.norm(xhat, axis=1)
    hdir = Path(args.m3_root) / f"seed{args.seed}" / "heads"
    starts = {"cold": None}
    for name, K in (("cp_K8_xhat", 8), ("cp_K16_xhat", 16), ("cp_K16_xstar", 16)):
        head = make_head("lista_cp", Ad, K, mu=1.0 / smax2); head.load_state_dict(torch.load(hdir / f"{name}.pt"))
        starts[name] = head_outputs(head, Yte, lam)
    t0 = time.perf_counter()
    RES = {}; E1 = dict(violations=0, checks=0); stats = {}
    for sname, x0 in starts.items():
        RES[sname] = {}
        for tau in TAUS:
            r = run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat, mode="rhoS", rho=rho)
            rel_err = np.linalg.norm(r.x - xhat, axis=1) / xn
            ent = dict(k_star_mean=float(r.k_star.mean()), k_star_std=float(r.k_star.std()), k_star_median=float(np.median(r.k_star)),
                       k_star_max=int(r.k_star.max()), nmse_out_db=nmse_db(r.x, Xte), time_s=r.time_s,
                       rel_err_max=float(rel_err.max()), guarantee_holds=bool(np.all(rel_err <= tau + 1e-9)),
                       tightness=percentiles(r.ratios[0]) if r.ratios else None, screening="shared, at x_{k-1}")
            kS = r.extra["k_S_equals_supp_xhat"]
            ent["frac_S_equals_supp_before_stop"] = float(np.mean((kS >= 0) & (kS <= r.k_star)))
            RES[sname][str(tau)] = ent
            E1["violations"] += r.violations; E1["checks"] += r.checks
            if tau == 1e-3:
                traces[f"kstar_ista_rhoS_{sname}"] = r.k_star
                if r.ratios:
                    traces[f"ratios_ista_rhoS_{sname}"] = r.ratios[0]
            if sname == "cold":
                rs = r.extra["rhoS_at_stop"]; Ssz = r.extra["S_size_at_stop"]; supp = (xhat != 0).sum(1)
                ok = np.isfinite(rs) & (Ssz > 0)
                stats[str(tau)] = dict(k_star_mean=float(r.k_star.mean()), rhoS_at_stop_median=float(np.median(rs[ok])),
                                       rhoS_at_stop_p90=float(np.percentile(rs[ok], 90)), frac_global_fallback=float(np.mean(rs[ok] >= rho - 1e-12)),
                                       S_size_median=float(np.median(Ssz[ok])), S_over_supp_median=float(np.median(Ssz[ok] / np.maximum(supp[ok], 1))),
                                       cert_factor_median=float(np.median(rs[ok] / (1 - rs[ok]))))
        print(f"ista_rhoS {sname:13s} k* " + " ".join(f"{t:g}:{RES[sname][str(t)]['k_star_mean']:6.1f}" for t in TAUS)
              + f"  viol {E1['violations']}  [{time.perf_counter() - t0:.0f}s]", flush=True)
    S["results"]["ista_rhoS"] = RES; S["E1"]["ista_rhoS"] = E1
    S.setdefault("notes", {})["ista_rhoS"] = "rerun 2026-10-22 with shared-product screening at x_{k-1} (one operator pair per iteration)"
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    np.savez(out / "traces.npz", **traces)
    old = json.load(open(out / "rhoS_stats.json"))
    oracle = np.array([rho_support(Ad, xhat[i] != 0, mu, eps) for i in range(min(args.n_oracle, Yte.shape[0]))])
    old.update(rho_S_oracle_median=float(np.median(oracle)), rho_S_oracle_p90=float(np.percentile(oracle, 90)), taus=stats,
               screening="shared, at x_{k-1}")
    json.dump(old, open(out / "rhoS_stats.json", "w"), indent=1)
    print(f"updated {out} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
