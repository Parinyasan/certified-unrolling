from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.config import config_hash, load_config
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import certificate_ratio, nmse_db, percentiles, theorem2_bound, worst_case_ratio
from cu.models.mondeq_tail import MonDEQTail, ZEncoder
from cu.train.train_mondeq import encoder_outputs, evaluate_mondeq, train_encoder, train_mondeq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=25600)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--m", type=float, default=0.5, help="strong monotonicity parameter")
    ap.add_argument("--L-max", type=float, default=1.5)
    ap.add_argument("--d", type=int, default=None, help="latent dimension (default n)")
    ap.add_argument("--eps-tols", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--out", default="results/m4")
    ap.add_argument("--reuse-from", default=None, help="load mondeq.pt/encoder.pt from this dir instead of training (revision re-certification)")
    ap.add_argument("--rel-tau", action="store_true", help="observable relative stopping rule C <= tau(||z_k|| - C) (revision)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    out = Path(args.out) / f"seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    tr_cfg = dict(steps=args.steps, batch=64, lr=1e-3, L_max=args.L_max, lam_L=10.0, max_iter=100, tol=1e-4)
    h = config_hash(dict(problem=cfg["problem"], train=tr_cfg, m=args.m, d=args.d, n_test=args.n_test, seed=args.seed))
    print(f"=== M4 Variant B  seed={args.seed} hash={h}  m={args.m} L_max={args.L_max} train={tr_cfg}")
    T_ALL = time.perf_counter()
    summary = dict(hash=h, seed=args.seed, m=args.m, L_max=args.L_max, train=tr_cfg, eps_tols=args.eps_tols, timing={})

    A, Ytr, Xtr, mtr = load_synthetic(cfg["problem"], args.n_train, seed=args.seed)
    _, Yte, Xte, mte = load_synthetic(cfg["problem"], args.n_test, seed=args.seed + 1000, A=A)
    _, Yva, Xva, mva = load_synthetic(cfg["problem"], 500, seed=args.seed + 2000, A=A)

    t0 = time.perf_counter()
    tail = MonDEQTail(m_y=A.m, n_out=A.n, d=args.d, m=args.m)
    if args.reuse_from:
        tail.load_state_dict(torch.load(Path(args.reuse_from) / "mondeq.pt")); hist = None
        summary["reused_from"] = args.reuse_from
        prev = Path(args.reuse_from) / "summary.json"
        if prev.exists():
            pv = json.load(open(prev)); summary["timing"]["train_tail_s"] = pv["timing"]["train_tail_s"]; summary["train_hist"] = pv.get("train_hist")
    else:
        hist = train_mondeq(tail, Ytr, Xtr, tr_cfg, Y_val=Yva, X_val=Xva, seed=args.seed)
        torch.save(tail.state_dict(), out / "mondeq.pt")
        summary["timing"]["train_tail_s"] = time.perf_counter() - t0
        summary["train_hist"] = {k: v for k, v in hist.items() if k != "total_time_s"}

    t0 = time.perf_counter()
    nt = tail.numpy_tail()
    rho_B = nt.rho()
    summary.update(L=nt.L, alpha=nt.alpha_, alpha_bound_formula=nt.alpha_bound_, rho_B=rho_B, rho_B_bound_formula=nt.rho_bound(),
                   worst_case_ratio=worst_case_ratio(rho_B),
                   strong_monotonicity_gap=nt.strong_monotonicity_gap(), C_norm=nt.C_norm, d=nt.M.shape[0])
    zstar_te, it_te, fd_te = nt.solve_fixed_point(Yte, tol=1e-12, max_iter=100_000)
    zstar_tr, it_tr, fd_tr = nt.solve_fixed_point(Ytr, tol=1e-10, max_iter=100_000)
    assert np.all(fd_te <= 1e-12) and np.all(fd_tr <= 1e-10)
    x_fp = nt.readout(zstar_te, Yte)
    summary["nmse_readout_xstar_db"] = nmse_db(x_fp, Xte)
    summary["nmse_readout_float32_forward_db"] = evaluate_mondeq(tail, Yte, Xte)
    summary["zstar_iters_mean"] = float(it_te.mean())
    summary["timing"]["fixed_points_s"] = time.perf_counter() - t0
    print(f"rho_B={rho_B:.5f} (L={nt.L:.3f}, alpha={nt.alpha_:.4f}); NMSE(readout(z*), x*) = {summary['nmse_readout_xstar_db']:.2f} dB; "
          f"z* iters mean {it_te.mean():.0f}  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    enc = ZEncoder(A.m, nt.M.shape[0])
    if args.reuse_from and (Path(args.reuse_from) / "encoder.pt").exists():
        enc.load_state_dict(torch.load(Path(args.reuse_from) / "encoder.pt"))
    else:
        train_encoder(enc, Ytr, zstar_tr, dict(steps=2000, batch=64, lr=1e-3), seed=args.seed)
        torch.save(enc.state_dict(), out / "encoder.pt")
    z0_w = encoder_outputs(enc, Yte)
    summary["timing"]["train_encoder_s"] = time.perf_counter() - t0
    znorm = np.linalg.norm(zstar_te, axis=1)
    summary["encoder_d_rel_median"] = float(np.median(np.linalg.norm(z0_w - zstar_te, axis=1) / znorm))

    t0 = time.perf_counter()
    slack = 10 * nt.certificate_factor() * 1e-12
    E1 = dict(violations=0, checks=0); E3 = {}; traces = {}; r_all = []
    for sname, z0 in (("cold", None), ("warm_encoder", z0_w)):
        d = np.linalg.norm((np.zeros_like(zstar_te) if z0 is None else z0) - zstar_te, axis=1)
        E3[sname] = {}
        for t in args.eps_tols:
            if args.rel_tau:
                res = nt.run_until(z0, Yte, 0.0, xhat=zstar_te, rel_tau=t); tol = t * znorm / (1 + 2 * t)
            else:
                tol = np.maximum(t * znorm, 1e-7 * np.linalg.norm(Yte, axis=1)); res = nt.run_until(z0, Yte, tol, xhat=zstar_te)
            assert np.all(res.converged)
            x_out = nt.readout(res.x, Yte)
            E3[sname][str(t)] = dict(k_star_mean=float(res.k_star.mean()), k_star_std=float(res.k_star.std()), k_star_max=int(res.k_star.max()),
                                     bound_mean=float(theorem2_bound(rho_B, d, tol).mean()),
                                     bound_dominates=int(np.sum(theorem2_bound(rho_B, d, tol) >= res.k_star)),
                                     nmse_out_db=nmse_db(x_out, Xte), tail_time_s=res.time_s,
                                     readout_err_bound_ok=bool(np.all(np.linalg.norm(x_out - x_fp, axis=1) <= nt.C_norm * (t * znorm if args.rel_tau else tol) + slack)))
            if t == min(args.eps_tols):
                E1["violations"] += int(np.sum(res.certs < res.errs_vs_xhat - slack)); E1["checks"] += int(res.certs.size)
                mask = np.arange(res.certs.shape[1])[None, :] < res.k_star[:, None]
                r_all.append(certificate_ratio(res.certs, res.errs_vs_xhat)[mask])
                traces[f"{sname}_certs"] = res.certs; traces[f"{sname}_errs"] = res.errs_vs_xhat; traces[f"{sname}_k_star"] = res.k_star
            traces[f"kstar_{sname}_{t}"] = res.k_star
        E3[sname]["d_rel_median"] = float(np.median(d / znorm))
    summary["E1"] = E1; summary["E3"] = E3; summary["E2"] = dict(r=percentiles(np.concatenate(r_all)))
    summary["timing"]["E1_E3_s"] = time.perf_counter() - t0
    print(f"E1-B: {E1['violations']} violations / {E1['checks']} checks; k* at 1e-3: cold {E3['cold']['0.001']['k_star_mean']:.1f}, "
          f"warm {E3['warm_encoder']['0.001']['k_star_mean']:.1f}; r_k median {summary['E2']['r']['p50']:.2f}  [{time.perf_counter()-t0:.1f}s]")

    m3 = Path(args.m3_root) / f"seed{args.seed}" / "summary.json"
    if m3.exists():
        a = json.load(open(m3))
        summary["compare"] = {t: dict(A_cold_k=a["E3"]["cold"][t]["k_star_mean"], A_warm_k=a["E3"]["cp_K8_xhat"][t]["k_star_mean"],
                                      A_nmse_db=a["E3"]["cp_K8_xhat"][t]["nmse_out_db"], A_tail_time_s=a["E3"]["cp_K8_xhat"][t]["tail_time_s"],
                                      B_cold_k=E3["cold"][t]["k_star_mean"], B_warm_k=E3["warm_encoder"][t]["k_star_mean"],
                                      B_nmse_db=E3["warm_encoder"][t]["nmse_out_db"], B_tail_time_s=E3["warm_encoder"][t]["tail_time_s"])
                              for t in map(str, args.eps_tols)}
        summary["compare"]["rho_A"] = a["rho"]; summary["compare"]["rho_B"] = rho_B
        summary["compare"]["A_nmse_xhat_db"] = a["nmse_xhat_xstar_db"]; summary["compare"]["B_nmse_fp_db"] = summary["nmse_readout_xstar_db"]

    summary["timing"]["total_s"] = time.perf_counter() - T_ALL
    np.savez_compressed(out / "traces.npz", **traces, rho_B=rho_B)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"TOTAL wall-clock seed {args.seed}: {summary['timing']['total_s']:.1f}s")


if __name__ == "__main__":
    main()
