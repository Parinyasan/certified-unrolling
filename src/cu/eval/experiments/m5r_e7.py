from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cu.algos.certificates import run_fista_gap, run_ista_certified, select_start
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.ista import ista
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.cwru import fault_band_energy_ratio, load_cwru
from cu.data.ecg import load_ecg
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db
from cu.models.lista import analytic_weight, make_head
from cu.ops.linear import DenseOperator
from cu.train.train_head import head_outputs


def perturb(A, rel, rng):
    if rel == 0:
        return A.copy()
    dA = rng.standard_normal(A.shape); dA *= rel * np.linalg.norm(A, 2) / np.linalg.norm(dA, 2)
    return A + dA


def alista_recompute(head, A_new):
    head.set_A(A_new)
    head.W = torch.as_tensor(analytic_weight(A_new, n_iter=500), dtype=head.W.dtype)


def certify(A2, Y, lam, eps_rel, tau, x0, xhat=None):
    tail = ContractiveTail(A2, lam, eps_rel * A2.sigma_max() ** 2)
    xh = xhat if xhat is not None else compute_and_cache_xhat(Y, A2, lam, tail.eps, mu=tail.mu, tol=1e-12, max_iter=200_000)[0]
    rb = run_ista_certified(A2, Y, lam, tail.eps, tail.mu, x0, tau, xhat=xh, mode="banach", rho=tail.rho())
    rf = run_fista_gap(A2, Y, lam, tail.eps, tail.mu, x0, tau, xhat=xh)
    return rb, rf, xh, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--shifts", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--e6-root", default="results/m5r/e6")
    ap.add_argument("--out", default="results/m5r/e7")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config); T0 = time.perf_counter()
    S = dict(shifts=args.shifts, tau=args.tau, seeds=args.seeds, synthetic={}, violations=0, checks=0)
    eps_rel = cfg["problem"]["eps_rel"]

    for seed in args.seeds:
        A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=seed); Ad = A.to_dense(); mu0 = 1.0 / A.sigma_max() ** 2
        hdir = Path(args.m3_root) / f"seed{seed}" / "heads"
        h8 = make_head("lista_cp", Ad, 8, mu=mu0); h8.load_state_dict(torch.load(hdir / "cp_K8_xhat.pt"))
        h16 = make_head("lista_cp", Ad, 16, mu=mu0); h16.load_state_dict(torch.load(hdir / "cp_K16_xstar.pt"))
        h16f = make_head("lista_cp", Ad, 16, mu=mu0); h16f.load_state_dict(torch.load(hdir / "cp_K16_xstar.pt"))
        al_path = hdir / "alista_K16_xstar.pt"
        al = make_head("alista", Ad, 16, mu=mu0)
        if al_path.exists():
            al.load_state_dict(torch.load(al_path))
        else:
            from cu.train.train_head import train_head
            _, Ytr, Xtr, mtr = load_synthetic(cfg["problem"], 25600, seed=seed)
            train_head(al, Ytr, Xtr, mtr["lam"], dict(batch=64, lr=5e-4, steps_per_stage=400, layerwise=True, loss="last"), seed=seed, log=lambda s: None)
            torch.save(al.state_dict(), al_path)
        alf = make_head("alista", Ad, 16, mu=mu0); alf.load_state_dict(torch.load(al_path))
        rng = np.random.default_rng(seed + 777)
        for rel in args.shifts:
            A2 = DenseOperator(perturb(Ad, rel, rng)); A2d = A2.to_dense()
            _, Y, X, meta = load_synthetic(cfg["problem"], args.n_test, seed=seed + 3000, A=A2); lam = meta["lam"]
            h8.set_A(A2d); h16.set_A(A2d); alista_recompute(al, A2d)
            x0 = head_outputs(h8, Y, lam)
            rb, rf, xh, tail = certify(A2, Y, lam, eps_rel, args.tau, x0)
            cb, cf, _, _ = certify(A2, Y, lam, eps_rel, args.tau, None, xhat=xh)
            S["violations"] += rb.violations + rf.violations + cb.violations + cf.violations; S["checks"] += rb.checks + rf.checks + cb.checks + cf.checks
            x_ista16 = ista(A2, Y, lam, 16, mu=1.0 / A2.sigma_max() ** 2)
            ent = dict(rho=tail.rho(), nmse_xhat_db=nmse_db(xh, X), ours_nmse_db=nmse_db(rb.x, X), ours_k=float(rb.k_star.mean()), ours_fista_k=float(rf.k_star.mean()),
                       cold_k=float(cb.k_star.mean()), cold_fista_k=float(cf.k_star.mean()),
                       lista16_adapt_nmse_db=nmse_db(head_outputs(h16, Y, lam), X), lista16_frozen_nmse_db=nmse_db(head_outputs(h16f, Y, lam), X),
                       alista16_recomputed_nmse_db=nmse_db(head_outputs(al, Y, lam), X), alista16_frozen_nmse_db=nmse_db(head_outputs(alf, Y, lam), X),
                       ista16_nmse_db=nmse_db(x_ista16, X), head8_d_rel=float(np.median(np.linalg.norm(x0 - xh, axis=1) / np.linalg.norm(xh, axis=1))))
            S["synthetic"].setdefault(str(rel), []).append(ent)
            print(f"seed {seed} shift {rel:4.2f}: xhat {ent['nmse_xhat_db']:6.2f} LISTA16 adapt {ent['lista16_adapt_nmse_db']:6.2f} frozen {ent['lista16_frozen_nmse_db']:6.2f} "
                  f"ALISTA recomp {ent['alista16_recomputed_nmse_db']:6.2f} frozen {ent['alista16_frozen_nmse_db']:6.2f} ISTA16 {ent['ista16_nmse_db']:6.2f} | k* ours {ent['ours_k']:.1f} cold {ent['cold_k']:.1f} fista {ent['ours_fista_k']:.1f}/{ent['cold_fista_k']:.1f}")

    S["ecg"] = {}
    e6 = Path(args.e6_root) / "ecg_compressive" / "seed0" / "heads"
    if (e6 / "lista_cp_K8_xhat.pt").exists():
        base = dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2)
        A0, _, _, _ = load_ecg(base, 5, 0, "test"); mu0 = 1.0 / A0.sigma_max() ** 2
        hc = make_head("lista_cp", A0.to_dense(), 8, mu=mu0); hc.load_state_dict(torch.load(e6 / "lista_cp_K8_xhat.pt"))
        ha = make_head("alista", A0.to_dense(), 8, mu=mu0) if (e6 / "alista_K8_xhat.pt").exists() else None
        if ha is not None:
            ha.load_state_dict(torch.load(e6 / "alista_K8_xhat.pt"))
        for ms in [0, 1, 2]:
            A2, Y, X, meta = load_ecg(dict(base, mask_seed=ms), args.n_test, 1000, "test"); A2d = A2.to_dense(); lam = meta["lam"]
            hc.set_A(A2d)
            sig = lambda Xc: float(10 * np.log10(np.mean(np.sum((Xc @ meta["Psi"].T - meta["S"]) ** 2, 1)) / np.mean(np.sum(meta["S"] ** 2, 1))))
            x0 = head_outputs(hc, Y, lam)
            rb, rf, xh, tail = certify(A2, Y, lam, 1e-2, args.tau, x0); cb, cf, _, _ = certify(A2, Y, lam, 1e-2, args.tau, None, xhat=xh)
            S["violations"] += rb.violations + rf.violations + cb.violations + cf.violations; S["checks"] += rb.checks + rf.checks + cb.checks + cf.checks
            ent = dict(same_matrix=(ms == 0), xhat_nmse_db=sig(xh), ours_nmse_db=sig(rb.x), ours_k=float(rb.k_star.mean()), cold_k=float(cb.k_star.mean()),
                       ours_fista_k=float(rf.k_star.mean()), cold_fista_k=float(cf.k_star.mean()), listacp_head_nmse_db=sig(x0),
                       ista8_nmse_db=sig(ista(A2, Y, lam, 8, mu=1.0 / A2.sigma_max() ** 2)))
            if ha is not None:
                alista_recompute(ha, A2d); xa = head_outputs(ha, Y, lam); ent["alista_recomputed_head_nmse_db"] = sig(xa)
                ra, _, _, _ = certify(A2, Y, lam, 1e-2, args.tau, xa, xhat=xh); ent["alista_ours_k"] = float(ra.k_star.mean())
            xs, keep, _, _ = select_start(A2, Y, x0, lam, tail.eps)
            rs, _, _, _ = certify(A2, Y, lam, 1e-2, args.tau, xs, xhat=xh)
            S["violations"] += rs.violations; S["checks"] += rs.checks
            ent["select_k"] = float(rs.k_star.mean()) + 1.0
            ent["select_frac_head_kept"] = float(keep.mean())
            S["ecg"][str(ms)] = ent
            print(f"ECG matrix {ms}: head LISTA-CP {ent['listacp_head_nmse_db']:.2f} ALISTA {ent.get('alista_recomputed_head_nmse_db', float('nan')):.2f} ISTA8 {ent['ista8_nmse_db']:.2f} ours {ent['ours_nmse_db']:.2f} dB; k* ours {ent['ours_k']:.1f} cold {ent['cold_k']:.1f}")

    S["cwru"] = {}
    c6 = Path(args.e6_root) / "cwru_or007" / "seed0" / "heads" / "lista_cp_K8_xhat.pt"
    if c6.exists():
        base = dict(file="130", n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2)
        A0, _, _, m0 = load_cwru(base, 5, 0, "test")
        hc = make_head("lista_cp", A0.to_dense(), 8, mu=1.0 / A0.sigma_max() ** 2); hc.load_state_dict(torch.load(c6))
        for fs_ in [1.0, 0.9, 1.1]:
            A2, Y, _, meta = load_cwru(dict(base, f_res=fs_ * m0["f_res"]), args.n_test, 1000, "test"); hc.set_A(A2.to_dense())
            x0 = head_outputs(hc, Y, meta["lam"])
            rb, rf, xh, tail = certify(A2, Y, meta["lam"], 1e-2, args.tau, x0); cb, cf, _, _ = certify(A2, Y, meta["lam"], 1e-2, args.tau, None, xhat=xh)
            S["violations"] += rb.violations + rf.violations + cb.violations + cf.violations; S["checks"] += rb.checks + rf.checks + cb.checks + cf.checks
            S["cwru"][str(fs_)] = dict(f_res=meta["f_res"], ours_k=float(rb.k_star.mean()), cold_k=float(cb.k_star.mean()), ours_fista_k=float(rf.k_star.mean()),
                                       fb_ours=float(fault_band_energy_ratio(rb.x, meta["f_fault"]).mean()), fb_head=float(fault_band_energy_ratio(x0, meta["f_fault"]).mean()))
            print(f"CWRU f_res x{fs_}: k* ours {S['cwru'][str(fs_)]['ours_k']:.1f} cold {S['cwru'][str(fs_)]['cold_k']:.1f}")

    sh = np.array(args.shifts)
    def agg(key):
        v = np.array([[e[key] for e in S["synthetic"][str(r)]] for r in args.shifts]); return v.mean(1), v.std(1)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for key, lab, st in (("ours_nmse_db", "certified (head + tail), any tail", "o-"), ("lista16_adapt_nmse_db", "LISTA-CP K=16 alone, coupling A'", "s--"),
                         ("lista16_frozen_nmse_db", "LISTA-CP K=16 alone, coupling A (frozen)", "d:"), ("alista16_recomputed_nmse_db", "ALISTA K=16 alone, W(A') recomputed", "^--"),
                         ("ista16_nmse_db", "16 ISTA steps (no learning)", "x-."), ("nmse_xhat_db", "elastic-net solution of A'", "k-")):
        m, s = agg(key); axes[0].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.1, capsize=2, label=lab)
    axes[0].set_xlabel(r"$\|\Delta A\|_2/\|A\|_2$ (%)"); axes[0].set_ylabel(r"NMSE vs $x^\star$ (dB)"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for key, lab, st in (("ours_k", "ISTA + Banach, K=8 head", "o-"), ("cold_k", "ISTA + Banach, cold", "s--"), ("ours_fista_k", "FISTA + gap, K=8 head", "^-"), ("cold_fista_k", "FISTA + gap, cold", "v--")):
        m, s = agg(key); axes[1].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.1, capsize=2, label=lab)
    axes[1].set_xlabel(r"$\|\Delta A\|_2/\|A\|_2$ (%)"); axes[1].set_ylabel(r"certified iterations $k^\ast$ ($\tau=10^{-3}$)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig8_operator_shift.png", dpi=170, bbox_inches="tight"); fig.savefig(out / "fig8_operator_shift.pdf", bbox_inches="tight")
    S["timing_total_s"] = time.perf_counter() - T0
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(f"E7r done: {S['violations']} violations / {S['checks']} checks; {S['timing_total_s']:.0f}s")


if __name__ == "__main__":
    main()
