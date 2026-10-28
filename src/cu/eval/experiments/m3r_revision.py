from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certificates import (GAP_FLOOR_C, coordinate_descent, gap_certificate, rho_support, rounding_guard,
                                   run_fista_gap, run_ista_certified)
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.ista import fista, ista
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db, percentiles, theorem2_bound
from cu.models.lista import make_head
from cu.train.train_head import evaluate, head_outputs

TAUS = [1e-2, 1e-3, 1e-4]


def float32_ablation(Ad, Y, lam, eps, mu, rho, xhat64, tau, guard, max_iter=5000):
    A32 = Ad.astype(np.float32); Y32 = Y.astype(np.float32); lam32 = lam.astype(np.float32)[:, None]
    mu32 = np.float32(mu); eps32 = np.float32(eps); fac = np.float32(rho / (1 - rho))
    x = np.zeros((Y.shape[0], Ad.shape[1]), np.float32)
    done = np.zeros(Y.shape[0], bool); viol = 0; checks = 0; kst = np.full(Y.shape[0], -1)
    for k in range(1, max_iter + 1):
        v = x - mu32 * ((x @ A32.T - Y32) @ A32)
        xn = (np.sign(v) * np.maximum(np.abs(v) - mu32 * lam32, 0)) / (1 + mu32 * eps32)
        c = fac * np.linalg.norm(xn - x, axis=1)
        if guard:
            c = c + rounding_guard(xn, x, rho, np.float32)
        err = np.linalg.norm(xn.astype(np.float64) - xhat64, axis=1)
        act = ~done
        viol += int(np.sum((c < err) & act)); checks += int(act.sum())
        fired = act & (c <= tau * (np.linalg.norm(xn.astype(np.float64), axis=1) - c))
        kst[fired] = k; done |= fired; x = xn
        if done.all():
            break
    kst[~done] = max_iter
    return dict(violations=viol, checks=checks, k_star_mean=float(kst.mean()), frac_not_fired=float(np.mean(~done)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--out", default="results/m3r")
    ap.add_argument("--eps-sweep", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--n-iter-trace", type=int, default=300)
    ap.add_argument("--no-extra", action="store_true", help="skip seed-0-only extras (ideal starts, diag heads)")
    args = ap.parse_args()
    out = Path(args.out) / f"seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config); T0 = time.perf_counter()
    S = dict(seed=args.seed, taus=TAUS, timing={})

    t0 = time.perf_counter()
    A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; lam = mte["lam"]; smax2 = A.sigma_max() ** 2
    tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
    mu1 = 1.0 / smax2; rho1 = ContractiveTail(A, lam, eps, mu=mu1).rho()
    xhat, info = compute_and_cache_xhat(Yte, A, lam, eps, mu=mu, tol=1e-12, verbose=True)
    xn = np.linalg.norm(xhat, axis=1)
    S.update(m=A.m, n=A.n, mu=mu, mu_times_smax2=mu * smax2, rho=rho, rho_old=rho1, eps=eps, nmse_xhat_xstar_db=nmse_db(xhat, Xte),
             xhat_iters_mean=float(info["n_iter"].mean()), xhat_time_s=info["time_s"])
    cd_rel = [float(np.linalg.norm(coordinate_descent(A, Yte[i], lam[i], eps) - xhat[i]) / xn[i]) for i in range(10)]
    S["cd_check_rel_diff_max"] = max(cd_rel)
    b_floor, _ = gap_certificate(A, Yte, xhat, lam, eps)
    S["gap_floor_rel_median"] = float(np.median(b_floor / xn)); S["gap_floor_C"] = GAP_FLOOR_C
    S["timing"]["setup_s"] = time.perf_counter() - t0
    print(f"mu = {mu*smax2:.2f}/smax^2, rho = {rho:.5f} (old {rho1:.5f}); CD check max rel diff {max(cd_rel):.1e}; gap floor {S['gap_floor_rel_median']:.1e} rel")

    hdir = Path(args.m3_root) / f"seed{args.seed}" / "heads"
    starts = {"cold": None}
    S["heads"] = {}
    for name, K in (("cp_K8_xhat", 8), ("cp_K16_xhat", 16), ("cp_K16_xstar", 16)):
        head = make_head("lista_cp", Ad, K, mu=1.0 / smax2); head.load_state_dict(torch.load(hdir / f"{name}.pt"))
        x0 = head_outputs(head, Yte, lam); starts[name] = x0
        S["heads"][name] = dict(K=K, d_rel_median=float(np.median(np.linalg.norm(x0 - xhat, axis=1) / xn)),
                                nmse_vs_xhat_db=nmse_db(x0, xhat), nmse_vs_xstar_db=nmse_db(x0, Xte),
                                per_layer_nmse_xstar=[evaluate(head, Yte, Xte, lam, n_layers=k + 1) for k in range(K)])

    t0 = time.perf_counter()
    RES = {}; traces = {}; E1 = {}
    methods = ["ista_banach", "ista_banach_mu1", "ista_gap", "ista_rhoS", "fista_gap"]
    for meth in methods:
        RES[meth] = {}; E1[meth] = dict(violations=0, checks=0)
        for sname, x0 in starts.items():
            if meth == "ista_banach_mu1" and sname not in ("cold", "cp_K16_xhat"):
                continue
            RES[meth][sname] = {}
            d = np.linalg.norm((0 if x0 is None else x0) - xhat, axis=1)
            for tau in TAUS:
                if meth == "ista_banach":
                    r = run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat, mode="banach", rho=rho)
                elif meth == "ista_banach_mu1":
                    r = run_ista_certified(A, Yte, lam, eps, mu1, x0, tau, xhat=xhat, mode="banach", rho=rho1)
                elif meth == "ista_gap":
                    r = run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat, mode="gap", rho=rho)
                elif meth == "ista_rhoS":
                    r = run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat, mode="rhoS", rho=rho)
                else:
                    r = run_fista_gap(A, Yte, lam, eps, mu, x0, tau, xhat=xhat)
                rel_err = np.linalg.norm(r.x - xhat, axis=1) / xn
                ent = dict(k_star_mean=float(r.k_star.mean()), k_star_std=float(r.k_star.std()), k_star_median=float(np.median(r.k_star)),
                           k_star_max=int(r.k_star.max()), nmse_out_db=nmse_db(r.x, Xte), time_s=r.time_s,
                           rel_err_max=float(rel_err.max()), guarantee_holds=bool(np.all(rel_err <= tau + 1e-9)),
                           tightness=percentiles(r.ratios[0]) if r.ratios else None)
                if meth in ("ista_banach", "ista_banach_mu1"):
                    rr = rho if meth == "ista_banach" else rho1
                    bound = theorem2_bound(rr, d, tau * xn / (1 + 2 * tau))
                    ent["bound_mean"] = float(bound.mean()); ent["bound_dominates"] = int(np.sum(bound >= r.k_star))
                if meth == "ista_rhoS":
                    kS = r.extra["k_S_equals_supp_xhat"]
                    ent["frac_S_equals_supp_before_stop"] = float(np.mean((kS >= 0) & (kS <= r.k_star)))
                RES[meth][sname][str(tau)] = ent
                E1[meth]["violations"] += r.violations; E1[meth]["checks"] += r.checks
                if tau == 1e-3:
                    traces[f"kstar_{meth}_{sname}"] = r.k_star
                    if r.ratios:
                        traces[f"ratios_{meth}_{sname}"] = r.ratios[0]
            print(f"{meth:16s} {sname:13s} k* " + " ".join(f"{t:g}:{RES[meth][sname][str(t)]['k_star_mean']:6.1f}" for t in TAUS)
                  + f"  viol {E1[meth]['violations']}  [{time.perf_counter()-t0:.0f}s]")
    S["results"] = RES; S["E1"] = E1
    S["timing"]["tails_s"] = time.perf_counter() - t0

    _, tr_i = ista(A, Yte, lam, args.n_iter_trace, mu=mu, return_trace=True)
    _, tr_f = fista(A, Yte, lam, args.n_iter_trace, mu=mu, return_trace=True)
    _, tr_e = ista(A, Yte, lam, args.n_iter_trace, mu=mu, eps=eps, return_trace=True)
    _, tr_fe = fista(A, Yte, lam, args.n_iter_trace, mu=mu, eps=eps, return_trace=True)
    S["E4"] = dict(ista=[nmse_db(x, Xte) for x in tr_i], fista=[nmse_db(x, Xte) for x in tr_f],
                   enista=[nmse_db(x, Xte) for x in tr_e], enfista=[nmse_db(x, Xte) for x in tr_fe])

    t0 = time.perf_counter(); E5 = {}
    x0_w = starts["cp_K8_xhat"]
    for eps_rel in args.eps_sweep:
        e = eps_rel * smax2; tl = ContractiveTail(A, lam, e)
        xh, inf = compute_and_cache_xhat(Yte, A, lam, e, mu=tl.mu, tol=1e-12, max_iter=200_000)
        ent = dict(rho=tl.rho(), nmse_xhat_db=nmse_db(xh, Xte), xhat_iters=float(inf["n_iter"].mean()), k=dict(), k_warm=dict(), k_fista=dict())
        for tau in TAUS:
            ent["k"][str(tau)] = float(run_ista_certified(A, Yte, lam, e, tl.mu, None, tau, mode="banach", rho=tl.rho()).k_star.mean())
            ent["k_warm"][str(tau)] = float(run_ista_certified(A, Yte, lam, e, tl.mu, x0_w, tau, mode="banach", rho=tl.rho()).k_star.mean())
            ent["k_fista"][str(tau)] = float(run_fista_gap(A, Yte, lam, e, tl.mu, None, tau).k_star.mean())
        ent["rho_S_oracle_median"] = float(np.median([rho_support(Ad, xh[i] != 0, tl.mu, e) for i in range(min(200, xh.shape[0]))]))
        E5[str(eps_rel)] = ent
        print(f"E5 eps_rel={eps_rel:g}: rho={ent['rho']:.4f} NMSE {ent['nmse_xhat_db']:.2f} k*(1e-3) cold {ent['k']['0.001']:.1f} warm {ent['k_warm']['0.001']:.1f} fista {ent['k_fista']['0.001']:.1f}")
    xl = ista(A, Yte, lam, 2000, mu=1.0 / smax2, x0=fista(A, Yte, lam, 20000, mu=1.0 / smax2))
    E5["0.0"] = dict(rho=1.0, nmse_xhat_db=nmse_db(xl, Xte), rho_S_oracle_median=float(np.median([rho_support(Ad, xl[i] != 0, 1.0 / smax2, 0.0) for i in range(min(200, xl.shape[0]))])))
    S["E5"] = E5; S["timing"]["E5_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    S["float32"] = {str(t): dict(no_guard=float32_ablation(Ad, Yte, lam, eps, mu, rho, xhat, t, False),
                                 guard=float32_ablation(Ad, Yte, lam, eps, mu, rho, xhat, t, True)) for t in TAUS}
    S["timing"]["float32_s"] = time.perf_counter() - t0
    print("float32: " + ", ".join(f"tau {t}: {v['no_guard']['violations']} -> {v['guard']['violations']} viol (guard), k* {v['guard']['k_star_mean']:.0f}" for t, v in S["float32"].items()))

    if args.seed == 0 and not args.no_extra:
        t0 = time.perf_counter(); rng = np.random.default_rng(1); ID = {}
        for rel in (1e-1, 1e-2, 1e-3):
            delta = rng.standard_normal(xhat.shape); delta *= rel * xn[:, None] / np.linalg.norm(delta, axis=1, keepdims=True)
            x0 = xhat + delta
            ID[str(rel)] = {str(t): dict(banach=float(run_ista_certified(A, Yte, lam, eps, mu, x0, t, mode="banach", rho=rho).k_star.mean()),
                                         fista_gap=float(run_fista_gap(A, Yte, lam, eps, mu, x0, t).k_star.mean())) for t in TAUS}
        S["ideal"] = ID
        DG = {}
        ddir = Path(args.m3_root) / "diag" / "heads"
        for name, kind, K in (("cp_K16_s1600", "lista_cp", 16), ("cpss_K16_s400", "lista_cpss", 16), ("cpss_K8_s400", "lista_cpss", 8), ("cpss_K16_s1600", "lista_cpss", 16)):
            f = ddir / f"{name}.pt"
            if not f.exists():
                continue
            head = make_head(kind, Ad, K, mu=1.0 / smax2); head.load_state_dict(torch.load(f))
            x0 = head_outputs(head, Yte, lam)
            DG[name] = dict(nmse_vs_xhat_db=nmse_db(x0, xhat), d_rel_median=float(np.median(np.linalg.norm(x0 - xhat, axis=1) / xn)),
                            k=dict((str(t), dict(banach=float(run_ista_certified(A, Yte, lam, eps, mu, x0, t, mode="banach", rho=rho).k_star.mean()),
                                                fista_gap=float(run_fista_gap(A, Yte, lam, eps, mu, x0, t).k_star.mean()))) for t in TAUS))
        S["diag_heads"] = DG; S["timing"]["extras_s"] = time.perf_counter() - t0
        print("ideal:", {k: v["0.001"] for k, v in ID.items()})

    S["timing"]["total_s"] = time.perf_counter() - T0
    np.savez_compressed(out / "traces.npz", **traces, xhat_norm=xn)
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(f"TOTAL seed {args.seed}: {S['timing']['total_s']:.0f}s")


if __name__ == "__main__":
    main()
