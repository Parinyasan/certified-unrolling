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
from cu.data.cwru import fault_band_energy_ratio, kurtosis, load_cwru
from cu.data.ecg import load_ecg
from cu.eval.metrics import certificate_ratio, nmse_db, percentiles, theorem2_bound, worst_case_ratio
from cu.models.lista import make_head
from cu.train.train_head import head_outputs, train_head

DATASETS = {
    "ecg_denoise_overcomplete": dict(kind="ecg", cfg=dict(dictionary="db4+dct", measurement="identity", snr_db=10.0, lam=0.02, eps_rel=1e-2)),
    "ecg_compressive": dict(kind="ecg", cfg=dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2)),
    "ecg_denoise_orthogonal": dict(kind="ecg", cfg=dict(dictionary="db4", measurement="identity", snr_db=10.0, lam=0.02, eps_rel=0.0)),
    "cwru_or007": dict(kind="cwru", cfg=dict(file="130", n_sig=1024, tau_ms=0.5, lam=0.1, eps_rel=1e-2)),
    "cwru_ir007": dict(kind="cwru", cfg=dict(file="105", n_sig=1024, tau_ms=0.5, lam=0.1, eps_rel=1e-2)),
}


def load(kind, cfg, n, seed, split):
    return load_ecg(cfg, n, seed, split) if kind == "ecg" else load_cwru(cfg, n, seed, split)


def signal_nmse_db(Xc, meta):
    S = meta["S"]
    return float(10 * np.log10(np.mean(np.sum((Xc @ meta["Psi"].T - S) ** 2, 1)) / np.mean(np.sum(S**2, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=6400)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eps-tols", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--out", default="results/m5/e6")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    for name in args.datasets:
        spec = DATASETS[name]; kind, dcfg = spec["kind"], spec["cfg"]
        out = Path(args.out) / name / f"seed{args.seed}"; (out / "heads").mkdir(parents=True, exist_ok=True)
        print(f"\n=== E6 {name} seed={args.seed}")
        T0 = time.perf_counter(); summ = dict(dataset=name, kind=kind, cfg=dcfg, seed=args.seed, K=args.K, timing={})
        t0 = time.perf_counter()
        A, Ytr, Xtr, mtr = load(kind, dcfg, args.n_train, args.seed, "train")
        _, Yte, Xte, mte = load(kind, dcfg, args.n_test, args.seed + 1000, "test")
        Ad = A.to_dense(); eps = mte["eps"]
        tail = ContractiveTail(A, mte["lam"], eps); rho = tail.rho(); mu = tail.mu
        summ.update(m=A.m, n=A.n, sigma_max=A.sigma_max(), sigma_min=A.sigma_min(), mu=mu, eps=eps, rho=rho, source=mte.get("source", kind),
                    n_train=Ytr.shape[0], n_test=Yte.shape[0])
        print(f"A {A.shape} smax={A.sigma_max():.3f} smin={A.sigma_min():.2e} rho={rho:.5f} mu={mu:.4f}")
        xhat_te, info_te = compute_and_cache_xhat(Yte, A, mte["lam"], eps, mu=mu, tol=1e-12, verbose=True, max_iter=200_000)
        xhat_tr, info_tr = compute_and_cache_xhat(Ytr, A, mtr["lam"], eps, mu=mu, tol=1e-10, verbose=True, max_iter=200_000)
        xhat_norm = np.linalg.norm(xhat_te, axis=1)
        summ["timing"]["data_xhat_s"] = time.perf_counter() - t0
        summ["xhat_iters_mean"] = float(info_te["n_iter"].mean())

        if kind == "ecg":
            summ["nmse_xhat_signal_db"] = signal_nmse_db(xhat_te, mte)
            summ["nmse_input_signal_db"] = float(10 * np.log10(np.mean(np.sum(mte["N"] ** 2, 1)) / np.mean(np.sum(mte["S"] ** 2, 1))))
            summ["nmse_xhat_coef_db"] = nmse_db(xhat_te, Xte)
        else:
            summ["fault_band_ratio_y"] = float(fault_band_energy_ratio(Yte, mte["f_fault"]).mean())
            summ["fault_band_ratio_xhat"] = float(fault_band_energy_ratio(xhat_te, mte["f_fault"]).mean())
            summ["kurtosis_y"] = float(kurtosis(Yte).mean()); summ["kurtosis_xhat"] = float(kurtosis(xhat_te).mean())
            rng0 = np.random.default_rng(0); xr = np.zeros_like(xhat_te)
            for i in range(len(xr)):
                idx = rng0.choice(xr.shape[1], max(1, int(np.sum(xhat_te[i] != 0))), replace=False); xr[i, idx] = rng0.standard_normal(len(idx))
            summ["fault_band_ratio_random_impulses"] = float(fault_band_energy_ratio(xr, mte["f_fault"]).mean())
            summ["f_res"] = mte["f_res"]; summ["f_fault"] = mte["f_fault"]

        t0 = time.perf_counter()
        head = make_head("lista_cp", Ad, args.K, mu=mu)
        f = out / "heads" / f"cp_K{args.K}_xhat.pt"
        if args.skip_train and f.exists():
            head.load_state_dict(torch.load(f))
        else:
            train_head(head, Ytr, xhat_tr, mtr["lam"], dict(batch=64, lr=5e-4, steps_per_stage=args.steps, layerwise=True, loss="last"),
                       seed=args.seed, log=lambda s: None)
            torch.save(head.state_dict(), f)
        x0 = head_outputs(head, Yte, mte["lam"])
        summ["timing"]["train_head_s"] = time.perf_counter() - t0
        summ["head"] = dict(nmse_vs_xhat_db=nmse_db(x0, xhat_te), d_rel_median=float(np.median(np.linalg.norm(x0 - xhat_te, axis=1) / xhat_norm)))
        if kind == "ecg":
            summ["head"]["nmse_signal_db"] = signal_nmse_db(x0, mte)
        else:
            summ["head"]["fault_band_ratio"] = float(fault_band_energy_ratio(x0, mte["f_fault"]).mean())

        t0 = time.perf_counter()
        slack = 10 * tail.certificate_factor() * 1e-12 + 1e-13
        E = {}; viol = 0; checks = 0; ex = {}
        for sname, xs in (("cold", None), ("warm", x0)):
            d = np.linalg.norm((0 if xs is None else xs) - xhat_te, axis=1)
            E[sname] = {}
            for t in args.eps_tols:
                tol = np.maximum(t * xhat_norm, 1e-7 * np.linalg.norm(Yte, axis=1))
                res = tail.run_until(xs, Yte, tol, xhat=xhat_te)
                assert np.all(res.converged)
                ent = dict(k_star_mean=float(res.k_star.mean()), k_star_std=float(res.k_star.std()), k_star_max=int(res.k_star.max()),
                           bound_mean=float(theorem2_bound(rho, d, tol).mean()) if rho > 0 else 1.0,
                           tail_time_s=res.time_s, time_per_instance_ms=1e3 * res.time_s / Yte.shape[0])
                if kind == "ecg":
                    ent["nmse_signal_db"] = signal_nmse_db(res.x, mte)
                else:
                    ent["fault_band_ratio"] = float(fault_band_energy_ratio(res.x, mte["f_fault"]).mean()); ent["kurtosis"] = float(kurtosis(res.x).mean())
                E[sname][str(t)] = ent
                if t == min(args.eps_tols):
                    viol += int(np.sum(res.certs < res.errs_vs_xhat - slack)); checks += int(res.certs.size)
                    mask = np.arange(res.certs.shape[1])[None, :] < res.k_star[:, None]
                    ent["tightness"] = percentiles(certificate_ratio(res.certs, res.errs_vs_xhat)[mask])
                    if sname == "warm":
                        i = 0
                        ex = dict(y=Yte[i], x_out=res.x[i], certs=res.certs[i, :res.k_star[i]], errs=res.errs_vs_xhat[i, :res.k_star[i]], k_star=int(res.k_star[i]), x0=x0[i], xhat=xhat_te[i])
        summ["E1"] = dict(violations=viol, checks=checks); summ["E3"] = E; summ["worst_case_ratio"] = worst_case_ratio(rho) if rho < 1 else None
        summ["timing"]["certified_runs_s"] = time.perf_counter() - t0
        summ["timing"]["total_s"] = time.perf_counter() - T0
        print(f"E1: {viol} viol / {checks};  k* cold/warm at 1e-3: {E['cold']['0.001']['k_star_mean']:.1f} / {E['warm']['0.001']['k_star_mean']:.1f}"
              + (f";  signal NMSE: input {summ['nmse_input_signal_db']:.2f}, xhat {summ['nmse_xhat_signal_db']:.2f}, head-only {summ['head']['nmse_signal_db']:.2f}, ours(1e-3) {E['warm']['0.001']['nmse_signal_db']:.2f} dB" if kind == "ecg"
                 else f";  fault-band ratio: y {summ['fault_band_ratio_y']:.3f}, xhat {summ['fault_band_ratio_xhat']:.3f}, head-only {summ['head']['fault_band_ratio']:.3f}")
              + f"  [total {summ['timing']['total_s']:.0f}s]")
        with open(out / "summary.json", "w") as fh:
            json.dump(summ, fh, indent=1, default=float)
        if kind == "ecg":
            ex.update(s=mte["S"][0], recon=ex["x_out"] @ mte["Psi"].T, noisy=mte["S"][0] + mte["N"][0], fs=mte["fs"])
        else:
            ex.update(fs=mte["fs"], f_fault=mte["f_fault"], atom=mte["atom"])
        np.savez_compressed(out / "example.npz", **ex)


if __name__ == "__main__":
    main()
