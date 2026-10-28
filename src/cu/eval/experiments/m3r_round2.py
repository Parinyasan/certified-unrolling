from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certificates import banach_guard, run_fista_gap, run_ista_certified
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.prox import elastic_net_prox
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db, percentiles
from cu.models.lista import make_head
from cu.train.train_head import head_outputs

TAUS = [1e-2, 1e-3, 1e-4]


def float32_ablation(Ad, Y, lam, eps, mu, rho, xhat64, tau, guard, max_iter=5000):
    A32 = Ad.astype(np.float32); Y32 = Y.astype(np.float32); lam32 = lam.astype(np.float32)[:, None]; lam64 = lam[:, None]
    mu32 = np.float32(mu); eps32 = np.float32(eps); fac = rho / (1 - rho); n = Ad.shape[1]
    x = np.zeros((Y.shape[0], n), np.float32)
    done = np.zeros(Y.shape[0], bool); viol = 0; checks = 0; kst = np.full(Y.shape[0], -1); e_rel = []; guard_rel = []
    for k in range(1, max_iter + 1):
        v = x - mu32 * ((x @ A32.T - Y32) @ A32)
        xn = (np.sign(v) * np.maximum(np.abs(v) - mu32 * lam32, 0)) / (1 + mu32 * eps32)
        d32 = np.linalg.norm(xn - x, axis=1).astype(np.float64)
        xn64 = xn.astype(np.float64)
        if guard:
            x64 = x.astype(np.float64)
            T64 = elastic_net_prox(x64 - mu * ((x64 @ Ad.T - Y) @ Ad), mu * lam64, mu * eps)
            e = np.linalg.norm(xn64 - T64, axis=1)
            c = banach_guard(d32, e, rho, n, np.float32)
            act0 = ~done
            if act0.any():
                e_rel.append(float(np.median(e[act0] / np.maximum(np.linalg.norm(T64[act0], axis=1), 1e-30))))
                guard_rel.append(float(np.median((e[act0] / (1 - rho)) / np.maximum(np.linalg.norm(xn64[act0], axis=1), 1e-30))))
        else:
            c = fac * d32
        err = np.linalg.norm(xn64 - xhat64, axis=1)
        act = ~done
        viol += int(np.sum((c < err) & act)); checks += int(act.sum())
        fired = act & (c <= tau * (np.linalg.norm(xn64, axis=1) - c))
        kst[fired] = k; done |= fired; x = xn
        if done.all():
            break
    kst[~done] = max_iter
    out = dict(violations=viol, checks=checks, k_star_mean=float(kst.mean()), frac_not_fired=float(np.mean(~done)))
    if guard:
        out["step_error_rel_median"] = float(np.median(e_rel)); out["guard_term_rel_median"] = float(np.median(guard_rel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--out", default="results/m3r")
    ap.add_argument("--skip-float32", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / f"seed{args.seed}"
    if not (out / "summary_round1.json").exists():
        shutil.copy(out / "summary.json", out / "summary_round1.json")
    S = json.load(open(out / "summary.json")); T0 = time.perf_counter()
    traces = dict(np.load(out / "traces.npz"))
    cfg = load_config(args.config)

    A, Ytr, _, mtr = load_synthetic(cfg["problem"], args.n_train, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; lam = mte["lam"]; smax2 = A.sigma_max() ** 2
    tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
    xhat, _ = compute_and_cache_xhat(Yte, A, lam, eps, mu=mu, tol=1e-12, verbose=False)
    xn = np.linalg.norm(xhat, axis=1)

    hdir = Path(args.m3_root) / f"seed{args.seed}" / "heads"
    starts = {"cold": None}; heads = {}
    for name, K in (("cp_K8_xhat", 8), ("cp_K16_xhat", 16), ("cp_K16_xstar", 16)):
        head = make_head("lista_cp", Ad, K, mu=1.0 / smax2); head.load_state_dict(torch.load(hdir / f"{name}.pt"))
        starts[name] = head_outputs(head, Yte, lam); heads[name] = head

    t0 = time.perf_counter()
    for meth in ("ista_gap", "fista_gap"):
        S["results"][meth] = {}; S["E1"][meth] = dict(violations=0, checks=0)
        for sname, x0 in starts.items():
            S["results"][meth][sname] = {}
            for tau in TAUS:
                r = (run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat, mode="gap", rho=rho) if meth == "ista_gap"
                     else run_fista_gap(A, Yte, lam, eps, mu, x0, tau, xhat=xhat))
                rel_err = np.linalg.norm(r.x - xhat, axis=1) / xn
                ent = dict(k_star_mean=float(r.k_star.mean()), k_star_std=float(r.k_star.std()), k_star_median=float(np.median(r.k_star)),
                           k_star_max=int(r.k_star.max()), nmse_out_db=nmse_db(r.x, Xte), time_s=r.time_s,
                           rel_err_max=float(rel_err.max()), guarantee_holds=bool(np.all(rel_err <= tau + 1e-9)),
                           tightness=percentiles(r.ratios[0]) if r.ratios else None, counts_operator_pairs=True)
                if meth == "fista_gap":
                    ent["k_iter_mean"] = float(r.extra["k_iter"].mean()); ent["extra_ops"] = int(r.extra["extra_ops"])
                S["results"][meth][sname][str(tau)] = ent
                S["E1"][meth]["violations"] += r.violations; S["E1"][meth]["checks"] += r.checks
                if tau == 1e-3:
                    traces[f"kstar_{meth}_{sname}"] = r.k_star
                    if r.ratios:
                        traces[f"ratios_{meth}_{sname}"] = r.ratios[0]
            print(f"{meth:10s} {sname:13s} k* (operator pairs) " + " ".join(f"{t:g}:{S['results'][meth][sname][str(t)]['k_star_mean']:6.1f}" for t in TAUS)
                  + f"  viol {S['E1'][meth]['violations']}  [{time.perf_counter()-t0:.0f}s]")
    S["timing"]["round2_gap_s"] = time.perf_counter() - t0

    for eps_rel, ent in S["E5"].items():
        if "k_fista" not in ent:
            continue
        e = float(eps_rel) * smax2; tl = ContractiveTail(A, lam, e)
        ent["k_fista"] = {str(t): float(run_fista_gap(A, Yte, lam, e, tl.mu, None, t).k_star.mean()) for t in TAUS}
    if "ideal" in S:
        rng = np.random.default_rng(1)
        for rel, d in S["ideal"].items():
            delta = rng.standard_normal(xhat.shape); delta *= float(rel) * xn[:, None] / np.linalg.norm(delta, axis=1, keepdims=True)
            for t in TAUS:
                d[str(t)]["fista_gap"] = float(run_fista_gap(A, Yte, lam, eps, mu, xhat + delta, t).k_star.mean())
    if "diag_heads" in S:
        ddir = Path(args.m3_root) / "diag" / "heads"
        for name, d in S["diag_heads"].items():
            kind, K = ("lista_cp" if name.startswith("cp_") else "lista_cpss"), int(name.split("_K")[1].split("_")[0])
            head = make_head(kind, Ad, K, mu=1.0 / smax2); head.load_state_dict(torch.load(ddir / f"{name}.pt"))
            x0 = head_outputs(head, Yte, lam)
            for t in TAUS:
                d["k"][str(t)]["fista_gap"] = float(run_fista_gap(A, Yte, lam, eps, mu, x0, t).k_star.mean())

    if not args.skip_float32:
        t0 = time.perf_counter()
        S["float32"] = {str(t): dict(no_guard=float32_ablation(Ad, Yte, lam, eps, mu, rho, xhat, t, False),
                                     guard=float32_ablation(Ad, Yte, lam, eps, mu, rho, xhat, t, True)) for t in TAUS}
        S["float32_guard"] = "banach_guard: (rho ||d|| (1+(n+3)u) + ||e_k||)/(1-rho), e_k measured in float64"
        S["timing"]["round2_float32_s"] = time.perf_counter() - t0
        for t, v in S["float32"].items():
            print(f"float32 tau {t}: no-guard viol {v['no_guard']['violations']} k* {v['no_guard']['k_star_mean']:.1f} | guard viol {v['guard']['violations']} "
                  f"k* {v['guard']['k_star_mean']:.1f} not fired {100*v['guard']['frac_not_fired']:.0f}%  step err {v['guard']['step_error_rel_median']:.1e} "
                  f"guard term {v['guard']['guard_term_rel_median']:.1e} rel")

    n_tr = min(1000, Ytr.shape[0]); Ytr1 = Ytr[:n_tr]; lam_tr = mtr["lam"][:n_tr]
    xhat_tr, _ = compute_and_cache_xhat(Ytr1, A, lam_tr, eps, mu=mu, tol=1e-12, verbose=False)
    S["head_gap"] = {}
    for name in ("cp_K8_xhat", "cp_K16_xhat"):
        x_tr = head_outputs(heads[name], Ytr1, lam_tr)
        d_tr = np.linalg.norm(x_tr - xhat_tr, axis=1) / np.linalg.norm(xhat_tr, axis=1)
        d_te = np.linalg.norm(starts[name] - xhat, axis=1) / xn
        S["head_gap"][name] = dict(d_rel_train_mean=float(d_tr.mean()), d_rel_test_mean=float(d_te.mean()),
                                   d_rel_train_median=float(np.median(d_tr)), d_rel_test_median=float(np.median(d_te)), n_train_eval=int(n_tr))
        print(f"head gap {name}: d/||xhat|| train {d_tr.mean():.4f} test {d_te.mean():.4f}")

    S["timing"]["round2_total_s"] = time.perf_counter() - T0
    np.savez_compressed(out / "traces.npz", **traces)
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(f"round 2 seed {args.seed} done in {S['timing']['round2_total_s']:.0f}s")


if __name__ == "__main__":
    main()
