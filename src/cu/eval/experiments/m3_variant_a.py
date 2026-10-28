from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.ista import fista, ista
from cu.algos.tail import ContractiveTail
from cu.config import config_hash, load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import certificate_ratio, nmse_db, percentiles, theorem2_bound, worst_case_ratio
from cu.models.lista import make_head
from cu.train.train_head import evaluate, head_outputs, train_head

SLACK12 = 1e-12


def support_local_rho(A: np.ndarray, xhat: np.ndarray, mu: float, eps: float) -> np.ndarray:
    out = np.ones(xhat.shape[0])
    for i in range(xhat.shape[0]):
        S = np.flatnonzero(xhat[i])
        if len(S) == 0:
            out[i] = 0.0
            continue
        w = np.linalg.eigvalsh(A[:, S].T @ A[:, S])
        out[i] = max(abs(1 - mu * w[0]), abs(1 - mu * w[-1])) / (1 + mu * eps)
    return out


def trace_with_support(tail: ContractiveTail, x0, Y, xhat, n_iter):
    x = np.zeros_like(xhat) if x0 is None else np.array(x0, dtype=np.float64)
    supp_hat = xhat != 0
    diffs, errs, same = [], [], []
    for _ in range(n_iter):
        xn = tail.step(x, Y)
        diffs.append(np.linalg.norm(xn - x, axis=1))
        errs.append(np.linalg.norm(xn - xhat, axis=1))
        same.append(np.all((xn != 0) == supp_hat, axis=1))
        x = xn
    return np.stack(diffs, 1), np.stack(errs, 1), np.stack(same, 1)


def float32_tail_violations(A, Y, lam, eps, mu, xhat64, eps_tol, max_iter=5000):
    A32 = A.astype(np.float32); Y32 = Y.astype(np.float32); lam32 = lam.astype(np.float32)[:, None]
    mu32 = np.float32(mu); eps32 = np.float32(eps)
    rho = ContractiveTail(A, lam, eps, mu).rho()
    fac = np.float32(rho / (1 - rho))
    x = np.zeros((Y32.shape[0], A.shape[1]), dtype=np.float32)
    done = np.zeros(Y.shape[0], bool); viol = 0; checks = 0; k_star = np.full(Y.shape[0], -1)
    c = np.zeros(Y.shape[0], np.float32)
    for k in range(1, max_iter + 1):
        g = (x @ A32.T - Y32) @ A32
        v = x - mu32 * g
        xn = (np.sign(v) * np.maximum(np.abs(v) - mu32 * lam32, 0)) / (1 + mu32 * eps32)
        c = fac * np.linalg.norm(xn - x, axis=1)
        err = np.linalg.norm(xn.astype(np.float64) - xhat64, axis=1)
        viol += int(np.sum(c < err)); checks += len(c)
        fired = (~done) & (c <= eps_tol)
        k_star[fired] = k; done |= fired
        x = xn
        if done.all():
            break
    k_star[~done] = max_iter
    return dict(violations=viol, checks=checks, k_star_mean=float(k_star.mean()), min_cert_reached=float(c.min()),
                frac_not_fired=float(np.mean(~done)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ks", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--xstar-Ks", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--eps-tols", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--eps-sweep", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--n-iter-trace", type=int, default=300)
    ap.add_argument("--out", default="results/m3")
    ap.add_argument("--skip-train", action="store_true", help="reuse heads saved in out/heads if present")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = dict(cfg["head"]["train"]); tr_cfg["n_train"] = args.n_train; tr_cfg["steps_per_stage"] = args.steps
    out = Path(args.out) / f"seed{args.seed}"; (out / "heads").mkdir(parents=True, exist_ok=True)
    h = config_hash(dict(problem=cfg["problem"], train=tr_cfg, Ks=args.Ks, xstar_Ks=args.xstar_Ks, n_test=args.n_test, seed=args.seed))
    print(f"=== M3 Variant A  seed={args.seed}  hash={h}  train={tr_cfg}")
    T_ALL = time.perf_counter()
    summary = dict(hash=h, seed=args.seed, config=args.config, train=tr_cfg, eps_tols=args.eps_tols, timing={})

    t0 = time.perf_counter()
    A, Ytr, Xtr, mtr = load_synthetic(cfg["problem"], args.n_train, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    _, Yva, Xva, mva = load_synthetic(cfg["problem"], 500, seed=args.seed + 2000, A=A)
    Ad = A.to_dense(); eps = mte["eps"]; mu = 1.0 / A.sigma_max() ** 2
    tail = ContractiveTail(A, mte["lam"], eps, mu); rho = tail.rho()
    xhat_te, info_te = compute_and_cache_xhat(Yte, A, mte["lam"], eps, mu=mu, tol=1e-12, verbose=True)
    xhat_tr, info_tr = compute_and_cache_xhat(Ytr, A, mtr["lam"], eps, mu=mu, tol=1e-10, verbose=True)
    xhat_va, _ = compute_and_cache_xhat(Yva, A, mva["lam"], eps, mu=mu, tol=1e-10)
    xhat_norm = np.linalg.norm(xhat_te, axis=1)
    summary.update(m=A.m, n=A.n, rho=rho, mu=mu, eps=eps, sigma_max=A.sigma_max(), worst_case_ratio=worst_case_ratio(rho),
                   nmse_xhat_xstar_db=nmse_db(xhat_te, Xte), xhat_test_iters_mean=float(info_te["n_iter"].mean()),
                   xhat_train_time_s=info_tr["time_s"], xhat_train_iters_mean=float(info_tr["n_iter"].mean()))
    summary["timing"]["data_xhat_s"] = time.perf_counter() - t0
    print(f"A {A.shape} rho={rho:.6f}  NMSE(xhat,x*)={summary['nmse_xhat_xstar_db']:.2f} dB  xhat train {info_tr['time_s']:.1f}s  [{time.perf_counter()-t0:.1f}s]")

    heads = {}
    specs = [(f"cp_K{K}_xhat", K, "xhat", "last") for K in args.Ks]
    specs += [(f"cp_K{K}_xstar", K, "xstar", "last") for K in args.xstar_Ks]
    specs += [("cp_K8_xhat_all", 8, "xhat", "all_layers")]
    summary["heads"] = {}
    for name, K, target, loss in specs:
        t0 = time.perf_counter()
        head = make_head("lista_cp", Ad, K, mu=mu)
        f = out / "heads" / f"{name}.pt"
        Ttr, Tva = (xhat_tr, xhat_va) if target == "xhat" else (Xtr, Xva)
        if args.skip_train and f.exists():
            head.load_state_dict(torch.load(f)); hist = None
        else:
            print(f"--- training {name} (K={K}, target={target}, loss={loss})")
            hist = train_head(head, Ytr, Ttr, mtr["lam"], dict(tr_cfg, loss=loss), Y_val=Yva, T_val=Tva, lam_val=mva["lam"],
                              seed=args.seed, log=lambda s: None)
            torch.save(head.state_dict(), f)
        heads[name] = (head, K, target)
        x0 = head_outputs(head, Yte, mte["lam"])
        d = np.linalg.norm(x0 - xhat_te, axis=1)
        summary["heads"][name] = dict(K=K, target=target, loss=loss, train_time_s=time.perf_counter() - t0,
                                      nmse_vs_xhat_db=nmse_db(x0, xhat_te), nmse_vs_xstar_db=nmse_db(x0, Xte),
                                      d_rel_median=float(np.median(d / xhat_norm)), d_rel_mean=float(np.mean(d / xhat_norm)),
                                      final_train_loss=(hist["loss"][-1] if hist else None))
        s = summary["heads"][name]
        print(f"{name}: NMSE vs xhat {s['nmse_vs_xhat_db']:.2f} dB, vs x* {s['nmse_vs_xstar_db']:.2f} dB, "
              f"d/||xhat|| median {s['d_rel_median']:.4f}  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    starts = {"cold": None}
    starts.update({name: head_outputs(hd, Yte, mte["lam"]) for name, (hd, K, tgt) in heads.items()})
    slack = 10 * tail.certificate_factor() * SLACK12
    E = dict(k_star={}, bound={}, nmse_out={}, d={}, violations=0, checks=0, r_all=[], time_tail_s={})
    tr_arrays = {}
    for sname, x0 in starts.items():
        d = np.linalg.norm((np.zeros_like(xhat_te) if x0 is None else x0) - xhat_te, axis=1)
        E["d"][sname] = d
        E["k_star"][sname], E["bound"][sname], E["nmse_out"][sname], E["time_tail_s"][sname] = {}, {}, {}, {}
        for tol_rel in args.eps_tols:
            eps_tol = np.maximum(tol_rel * xhat_norm, 1e-7 * np.linalg.norm(Yte, axis=1))
            res = tail.run_until(x0, Yte, eps_tol, xhat=xhat_te)
            assert np.all(res.converged)
            E["k_star"][sname][tol_rel] = res.k_star
            E["bound"][sname][tol_rel] = theorem2_bound(rho, d, eps_tol)
            E["nmse_out"][sname][tol_rel] = nmse_db(res.x, Xte)
            E["time_tail_s"][sname][tol_rel] = res.time_s
            if tol_rel == min(args.eps_tols):
                v = int(np.sum(res.certs < res.errs_vs_xhat - slack)); E["violations"] += v; E["checks"] += int(res.certs.size)
                mask = np.arange(res.certs.shape[1])[None, :] < res.k_star[:, None]
                E["r_all"].append(certificate_ratio(res.certs, res.errs_vs_xhat)[mask])
                if sname in ("cold", "cp_K8_xhat"):
                    tr_arrays[f"{sname}_certs"] = res.certs; tr_arrays[f"{sname}_errs"] = res.errs_vs_xhat
                    tr_arrays[f"{sname}_diffs"] = res.diffs; tr_arrays[f"{sname}_k_star"] = res.k_star
    summary["E1"] = dict(violations=E["violations"], checks=E["checks"])
    summary["E3"] = {s: {str(t): dict(k_star_mean=float(E["k_star"][s][t].mean()), k_star_std=float(E["k_star"][s][t].std()),
                                       k_star_median=float(np.median(E["k_star"][s][t])), k_star_max=int(E["k_star"][s][t].max()),
                                       bound_mean=float(E["bound"][s][t].mean()),
                                       bound_dominates=int(np.sum(E["bound"][s][t] >= E["k_star"][s][t])),
                                       nmse_out_db=E["nmse_out"][s][t], tail_time_s=E["time_tail_s"][s][t])
                         for t in args.eps_tols} for s in starts}
    summary["E3"]["d_rel_median"] = {s: float(np.median(E["d"][s] / xhat_norm)) for s in starts}
    summary["timing"]["E1_E3_s"] = time.perf_counter() - t0
    print(f"E1: {E['violations']} violations / {E['checks']} checks.  E3 k* (mean) at tol 1e-3: " +
          ", ".join(f"{s}={E['k_star'][s][1e-3].mean():.1f}" for s in starts) + f"  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    diffs, errs, same = trace_with_support(tail, None, Yte, xhat_te, int(tr_arrays["cold_k_star"].max()))
    certs = (rho / (1 - rho)) * diffs
    kmask = np.arange(diffs.shape[1])[None, :] < tr_arrays["cold_k_star"][:, None]
    r_cold = certificate_ratio(certs, errs)[kmask]
    rho_hat = np.full_like(diffs, np.nan); rho_hat[:, 1:] = diffs[:, 1:] / diffs[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cert_hat = np.where(rho_hat < 1, rho_hat / (1 - rho_hat) * diffs, np.inf)
    ok_hat = kmask & np.isfinite(rho_hat)
    viol_hat = int(np.sum((cert_hat < errs) & ok_hat)); r_hat = certificate_ratio(cert_hat, errs)[ok_hat & np.isfinite(cert_hat)]
    rho_S = support_local_rho(Ad, xhat_te, mu, eps)
    cert_S = (rho_S / (1 - rho_S))[:, None] * diffs
    ok_S = kmask & same & (rho_S[:, None] < 1)
    viol_S = int(np.sum((cert_S < errs) & ok_S)); r_S = certificate_ratio(cert_S, errs)[ok_S]
    k_supp = np.array([int(np.argmax(same[i])) if same[i].any() else -1 for i in range(len(same))])
    summary["E2"] = dict(r_cold=percentiles(r_cold), r_all_starts=percentiles(np.concatenate(E["r_all"])),
                         worst_case=worst_case_ratio(rho), frac_r_above_worst_case=float(np.mean(r_cold > worst_case_ratio(rho))),
                         rho_hat=dict(violations=viol_hat, checks=int(ok_hat.sum()), ratio=percentiles(r_hat)),
                         rho_S=dict(violations=viol_S, checks=int(ok_S.sum()), ratio=percentiles(r_S),
                                    rho_S_median=float(np.median(rho_S)), rho_S_p90=float(np.percentile(rho_S, 90)),
                                    k_support_stable_median=float(np.median(k_supp[k_supp >= 0])) if (k_supp >= 0).any() else None,
                                    frac_support_stable_before_kstar=float(np.mean((k_supp >= 0) & (k_supp < tr_arrays["cold_k_star"])))))
    summary["timing"]["E2_s"] = time.perf_counter() - t0
    print(f"E2: r_k median {summary['E2']['r_cold']['p50']:.1f} p90 {summary['E2']['r_cold']['p90']:.1f} max {summary['E2']['r_cold']['p100']:.1f} "
          f"(worst-case {worst_case_ratio(rho):.0f}); rho_hat viol {viol_hat}/{ok_hat.sum()}; rho_S median {np.median(rho_S):.4f} "
          f"viol {viol_S}/{ok_S.sum()}  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    _, tr_i = ista(A, Yte, mte["lam"], args.n_iter_trace, return_trace=True)
    _, tr_f = fista(A, Yte, mte["lam"], args.n_iter_trace, return_trace=True)
    _, tr_e = ista(A, Yte, mte["lam"], args.n_iter_trace, eps=eps, return_trace=True)
    E4 = dict(ista=[nmse_db(x, Xte) for x in tr_i], fista=[nmse_db(x, Xte) for x in tr_f], enista=[nmse_db(x, Xte) for x in tr_e])
    for name, (hd, K, tgt) in heads.items():
        E4[f"{name}_per_layer"] = [evaluate(hd, Yte, Xte, mte["lam"], n_layers=k + 1) for k in range(K)]
    E4["ours"] = {s: {str(t): dict(ops=float((0 if s == "cold" else heads[s][1]) + E["k_star"][s][t].mean()), nmse_db=E["nmse_out"][s][t])
                      for t in args.eps_tols} for s in starts}
    summary["E4"] = E4; summary["timing"]["E4_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    E5 = {}
    x0_w = head_outputs(heads["cp_K8_xhat"][0], Yte, mte["lam"])
    for eps_rel in args.eps_sweep:
        e = eps_rel * A.sigma_max() ** 2
        entry = dict(eps_rel=eps_rel)
        if eps_rel > 0:
            tl = ContractiveTail(A, mte["lam"], e, mu)
            xh, inf = compute_and_cache_xhat(Yte, A, mte["lam"], e, mu=mu, tol=1e-12, max_iter=200_000)
            entry.update(rho=tl.rho(), nmse_xhat_db=nmse_db(xh, Xte), xhat_iters_mean=float(inf["n_iter"].mean()), k_star={}, k_star_warm={})
            for t in args.eps_tols:
                tol = np.maximum(t * np.linalg.norm(xh, axis=1), 1e-7 * np.linalg.norm(Yte, axis=1))
                rc = tl.run_until(None, Yte, tol); rw = tl.run_until(x0_w, Yte, tol)
                entry["k_star"][str(t)] = float(rc.k_star.mean()); entry["k_star_warm"][str(t)] = float(rw.k_star.mean())
            entry["rho_S_median"] = float(np.median(support_local_rho(Ad, xh, mu, e)))
        else:
            xl = fista(A, Yte, mte["lam"], 20000)
            xl = ista(A, Yte, mte["lam"], 2000, mu=mu, x0=xl)
            rS = support_local_rho(Ad, xl, mu, 0.0)
            entry.update(rho=1.0, nmse_xhat_db=nmse_db(xl, Xte), rho_S_median=float(np.median(rS)), rho_S_p90=float(np.percentile(rS, 90)),
                         frac_rho_S_below_1=float(np.mean(rS < 1)), k_star=None, k_star_warm=None)
        E5[str(eps_rel)] = entry
        print(f"E5 eps_rel={eps_rel:g}: rho={entry['rho']:.5f} NMSE(xhat,x*)={entry['nmse_xhat_db']:.2f} dB rho_S~{entry['rho_S_median']:.4f} "
              + (f"k*(1e-3) cold={entry['k_star']['0.001']:.1f} warm={entry['k_star_warm']['0.001']:.1f}" if entry["k_star"] else ""))
    summary["E5"] = E5; summary["timing"]["E5_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary["float32_ablation"] = {str(t): float32_tail_violations(Ad, Yte, mte["lam"], eps, mu, xhat_te,
                                                                   np.maximum(t * xhat_norm, 1e-7 * np.linalg.norm(Yte, axis=1)))
                                   for t in args.eps_tols}
    summary["timing"]["float32_s"] = time.perf_counter() - t0
    print("float32 tail: " + ", ".join(f"tol {t}: {v['violations']}/{v['checks']} viol, not fired {v['frac_not_fired']:.2f}"
                                       for t, v in summary["float32_ablation"].items()))

    summary["timing"]["total_s"] = time.perf_counter() - T_ALL
    np.savez_compressed(out / "traces.npz", **tr_arrays, rho=rho, xhat_norm=xhat_norm,
                        **{f"kstar_{s}_{t}": E["k_star"][s][t] for s in starts for t in args.eps_tols},
                        **{f"bound_{s}_{t}": E["bound"][s][t] for s in starts for t in args.eps_tols},
                        **{f"d_{s}": E["d"][s] for s in starts}, rho_S=rho_S, k_supp=k_supp, r_cold=r_cold, r_hat=r_hat, r_S=r_S)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"TOTAL wall-clock seed {args.seed}: {summary['timing']['total_s']:.1f}s")


if __name__ == "__main__":
    main()
