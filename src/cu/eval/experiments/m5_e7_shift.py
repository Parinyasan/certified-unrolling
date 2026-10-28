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

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.cwru import damped_sinusoid_atom, fault_band_energy_ratio, load_cwru
from cu.data.ecg import load_ecg
from cu.data.synthetic import load_synthetic
from cu.eval.metrics import nmse_db
from cu.models.lista import make_head
from cu.ops.linear import ConvOperator, DenseOperator
from cu.train.train_head import head_outputs


def perturb(A: np.ndarray, rel: float, rng) -> np.ndarray:
    if rel == 0:
        return A.copy()
    dA = rng.standard_normal(A.shape)
    dA *= rel * np.linalg.norm(A, 2) / np.linalg.norm(dA, 2)
    return A + dA


def certified(head, A_new, Y, lam, eps_rel, tol_rel, xhat=None):
    tail = ContractiveTail(A_new, lam, eps_rel * A_new.sigma_max() ** 2)
    xh = xhat if xhat is not None else compute_and_cache_xhat(Y, A_new, lam, tail.eps, mu=tail.mu, tol=1e-12, max_iter=200_000)[0]
    x0 = None if head is None else head_outputs(head, Y, lam)
    res = tail.run_until(x0, Y, np.maximum(tol_rel * np.linalg.norm(xh, axis=1), 1e-7 * np.linalg.norm(Y, axis=1)), xhat=xh)
    viol = int(np.sum(res.certs < res.errs_vs_xhat - 10 * tail.certificate_factor() * 1e-12))
    return res, xh, tail, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_base.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--shifts", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--m3-root", default="results/m3")
    ap.add_argument("--e6-root", default="results/m5/e6")
    ap.add_argument("--out", default="results/m5/e7")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config); T0 = time.perf_counter()
    S = dict(shifts=args.shifts, tol_rel=args.tol_rel, seeds=args.seeds, synthetic={}, violations=0, checks=0)

    for seed in args.seeds:
        A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=seed)
        Ad = A.to_dense(); mu = 1.0 / A.sigma_max() ** 2
        hdir = Path(args.m3_root) / f"seed{seed}" / "heads"
        head8 = make_head("lista_cp", Ad, 8, mu=mu); head8.load_state_dict(torch.load(hdir / "cp_K8_xhat.pt"))
        head16 = make_head("lista_cp", Ad, 16, mu=mu); head16.load_state_dict(torch.load(hdir / "cp_K16_xstar.pt"))
        head16_frozen = make_head("lista_cp", Ad, 16, mu=mu); head16_frozen.load_state_dict(torch.load(hdir / "cp_K16_xstar.pt"))
        rng = np.random.default_rng(seed + 777)
        for rel in args.shifts:
            A2 = DenseOperator(perturb(Ad, rel, rng))
            _, Y, X, meta = load_synthetic(cfg["problem"], args.n_test, seed=seed + 3000, A=A2)
            lam = meta["lam"]
            head8.set_A(A2.to_dense()); head16.set_A(A2.to_dense())
            res_w, xh, tail, v1 = certified(head8, A2, Y, lam, cfg["problem"]["eps_rel"], args.tol_rel)
            res_c, _, _, v2 = certified(None, A2, Y, lam, cfg["problem"]["eps_rel"], args.tol_rel, xhat=xh)
            S["violations"] += v1 + v2; S["checks"] += int(res_w.certs.size + res_c.certs.size)
            x_lista = head_outputs(head16, Y, lam); x_frozen = head_outputs(head16_frozen, Y, lam)
            x_head8 = head_outputs(head8, Y, lam)
            ent = dict(rho=tail.rho(), nmse_xhat_db=nmse_db(xh, X), ours_nmse_db=nmse_db(res_w.x, X), ours_k=float(res_w.k_star.mean()),
                       cold_k=float(res_c.k_star.mean()), lista16_adapt_nmse_db=nmse_db(x_lista, X), lista16_frozen_nmse_db=nmse_db(x_frozen, X),
                       head8_only_nmse_db=nmse_db(x_head8, X), head8_d_rel=float(np.median(np.linalg.norm(x_head8 - xh, axis=1) / np.linalg.norm(xh, axis=1))))
            S["synthetic"].setdefault(str(rel), []).append(ent)
            print(f"seed {seed} shift {rel:5.2f}: rho={ent['rho']:.4f} NMSE: xhat {ent['nmse_xhat_db']:6.2f} ours {ent['ours_nmse_db']:6.2f} "
                  f"LISTA16(adapt) {ent['lista16_adapt_nmse_db']:6.2f} LISTA16(frozen) {ent['lista16_frozen_nmse_db']:6.2f} dB | k*: ours {ent['ours_k']:.1f} cold {ent['cold_k']:.1f}")

    if not args.skip_real:
        S["ecg_mask"] = {}
        e6 = Path(args.e6_root) / "ecg_compressive" / "seed0" / "heads" / "cp_K8_xhat.pt"
        if e6.exists():
            base = dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2)
            A0, _, _, _ = load_ecg(base, 5, 0, "test")
            head = make_head("lista_cp", A0.to_dense(), 8, mu=1.0 / A0.sigma_max() ** 2); head.load_state_dict(torch.load(e6))
            for mask_seed in [0, 1, 2]:
                A2, Y, X, meta = load_ecg(dict(base, mask_seed=mask_seed), args.n_test, 1000, "test")
                head.set_A(A2.to_dense())
                res_w, xh, tail, v1 = certified(head, A2, Y, meta["lam"], 1e-2, args.tol_rel)
                res_c, _, _, v2 = certified(None, A2, Y, meta["lam"], 1e-2, args.tol_rel, xhat=xh)
                S["violations"] += v1 + v2; S["checks"] += int(res_w.certs.size + res_c.certs.size)
                sig = lambda Xc: float(10 * np.log10(np.mean(np.sum((Xc @ meta["Psi"].T - meta["S"]) ** 2, 1)) / np.mean(np.sum(meta["S"] ** 2, 1))))
                S["ecg_mask"][str(mask_seed)] = dict(same_mask=(mask_seed == 0), ours_k=float(res_w.k_star.mean()), cold_k=float(res_c.k_star.mean()),
                                                     ours_nmse_signal_db=sig(res_w.x), head_only_nmse_signal_db=sig(head_outputs(head, Y, meta["lam"])), xhat_nmse_signal_db=sig(xh))
                print(f"ECG mask {mask_seed}: k* ours {res_w.k_star.mean():.1f} cold {res_c.k_star.mean():.1f}; signal NMSE ours {S['ecg_mask'][str(mask_seed)]['ours_nmse_signal_db']:.2f} head-only {S['ecg_mask'][str(mask_seed)]['head_only_nmse_signal_db']:.2f} dB")
        S["cwru_atom"] = {}
        c6 = Path(args.e6_root) / "cwru_or007" / "seed0" / "heads" / "cp_K8_xhat.pt"
        if c6.exists():
            base = dict(file="130", n_sig=1024, tau_ms=0.5, lam=0.1, eps_rel=1e-2)
            A0, _, _, m0 = load_cwru(base, 5, 0, "test")
            head = make_head("lista_cp", A0.to_dense(), 8, mu=1.0 / A0.sigma_max() ** 2); head.load_state_dict(torch.load(c6))
            for fscale in [1.0, 0.9, 1.1]:
                A2, Y, _, meta = load_cwru(dict(base, f_res=fscale * m0["f_res"]), args.n_test, 1000, "test")
                head.set_A(A2.to_dense())
                res_w, xh, tail, v1 = certified(head, A2, Y, meta["lam"], 1e-2, args.tol_rel)
                res_c, _, _, v2 = certified(None, A2, Y, meta["lam"], 1e-2, args.tol_rel, xhat=xh)
                S["violations"] += v1 + v2; S["checks"] += int(res_w.certs.size + res_c.certs.size)
                S["cwru_atom"][str(fscale)] = dict(f_res=meta["f_res"], ours_k=float(res_w.k_star.mean()), cold_k=float(res_c.k_star.mean()),
                                                   ours_fault_ratio=float(fault_band_energy_ratio(res_w.x, meta["f_fault"]).mean()),
                                                   head_only_fault_ratio=float(fault_band_energy_ratio(head_outputs(head, Y, meta["lam"]), meta["f_fault"]).mean()))
                print(f"CWRU f_res x{fscale}: k* ours {res_w.k_star.mean():.1f} cold {res_c.k_star.mean():.1f}; fault ratio ours {S['cwru_atom'][str(fscale)]['ours_fault_ratio']:.3f}")

    sh = np.array(args.shifts)
    def agg(key):
        v = np.array([[e[key] for e in S["synthetic"][str(r)]] for r in args.shifts]); return v.mean(1), v.std(1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for key, lab, st in (("ours_nmse_db", "ours (K=8 head + certified tail)", "o-"), ("lista16_adapt_nmse_db", "LISTA-CP K=16 alone (coupling A')", "s--"),
                         ("lista16_frozen_nmse_db", "LISTA-CP K=16 alone (coupling A, frozen)", "d:"), ("nmse_xhat_db", "$\\hat x_\\varepsilon(A')$ (tail fixed point)", "k-.")):
        m, s = agg(key); axes[0].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.2, capsize=2, label=lab)
    axes[0].set_xlabel("$\\|\\Delta A\\|_2/\\|A\\|_2$ (%)"); axes[0].set_ylabel("NMSE vs $x^\\star$ (dB)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6.5)
    for key, lab, st in (("ours_k", "warm (K=8 head trained on A)", "o-"), ("cold_k", "cold start", "s--")):
        m, s = agg(key); axes[1].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.2, capsize=2, label=lab)
    axes[1].set_xlabel("$\\|\\Delta A\\|_2/\\|A\\|_2$ (%)"); axes[1].set_ylabel(f"certified tail iterations $k^\\ast$ ($\\varepsilon_{{\\rm tol}}=10^{{{int(np.log10(args.tol_rel))}}}$)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig8_operator_shift.png", dpi=170); fig.savefig(out / "fig8_operator_shift.pdf")
    S["timing_total_s"] = time.perf_counter() - T0
    with open(out / "summary.json", "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(f"E7 done: {S['violations']} violations / {S['checks']} checks; {S['timing_total_s']:.0f}s")


if __name__ == "__main__":
    main()
