from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certificates import run_fista_gap, run_ista_certified
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.data.cwru import fault_band_energy_ratio, kurtosis, load_cwru
from cu.data.ecg import load_ecg
from cu.eval.metrics import nmse_db, percentiles, theorem2_bound
from cu.models.lista import make_head
from cu.train.train_head import head_outputs, train_head

TAUS = [1e-2, 1e-3, 1e-4]
DATASETS = {
    "ecg_compressive": dict(kind="ecg", cfg=dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2)),
    "ecg_denoise_overcomplete": dict(kind="ecg", cfg=dict(dictionary="db4+dct", measurement="identity", snr_db=10.0, lam=0.02, eps_rel=1e-2)),
    "cwru_or007": dict(kind="cwru", cfg=dict(file="130", n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2)),
    "cwru_ir007": dict(kind="cwru", cfg=dict(file="105", n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2)),
}


def load(kind, cfg, n, seed, split):
    return load_ecg(cfg, n, seed, split) if kind == "ecg" else load_cwru(cfg, n, seed, split)


def sig_nmse(Xc, meta):
    S = meta["S"]
    return float(10 * np.log10(np.mean(np.sum((Xc @ meta["Psi"].T - S) ** 2, 1)) / np.mean(np.sum(S**2, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ecg_compressive", "cwru_or007", "cwru_ir007"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=6400)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--heads", nargs="+", default=["lista_cp", "alista", "conv_lista"])
    ap.add_argument("--out", default="results/m5r/e6")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    for name in args.datasets:
        spec = DATASETS[name]; kind, dcfg = spec["kind"], spec["cfg"]
        for seed in args.seeds:
            out = Path(args.out) / name / f"seed{seed}"; (out / "heads").mkdir(parents=True, exist_ok=True)
            print(f"\n=== E6r {name} seed={seed}"); T0 = time.perf_counter()
            summ = dict(dataset=name, kind=kind, cfg=dcfg, seed=seed, K=args.K, steps=args.steps, taus=TAUS, timing={})
            t0 = time.perf_counter()
            A, Ytr, Xtr, mtr = load(kind, dcfg, args.n_train, seed, "train")
            _, Yte, Xte, mte = load(kind, dcfg, args.n_test, seed + 1000, "test")
            Ad = A.to_dense(); eps = mte["eps"]; lam = mte["lam"]
            tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
            summ.update(m=A.m, n=A.n, sigma_max=A.sigma_max(), mu_times_smax2=mu * A.sigma_max() ** 2, rho=rho, eps=eps,
                        n_train=int(Ytr.shape[0]), n_test=int(Yte.shape[0]), source=str(mte.get("source", kind)))
            xhat_te, info_te = compute_and_cache_xhat(Yte, A, lam, eps, mu=mu, tol=1e-12, verbose=True, max_iter=200_000)
            xhat_tr, info_tr = compute_and_cache_xhat(Ytr, A, mtr["lam"], eps, mu=mu, tol=1e-10, verbose=True, max_iter=200_000)
            xn = np.linalg.norm(xhat_te, axis=1)
            summ["timing"]["data_xhat_s"] = time.perf_counter() - t0
            summ["offline_xhat_train_s"] = info_tr["time_s"]; summ["xhat_iters_mean"] = float(info_te["n_iter"].mean())
            if kind == "ecg":
                summ["metric"] = "signal_nmse_db"
                summ["nmse_input_db"] = float(10 * np.log10(np.mean(np.sum(mte["N"] ** 2, 1)) / np.mean(np.sum(mte["S"] ** 2, 1))))
                summ["nmse_xhat_db"] = sig_nmse(xhat_te, mte)
                met = lambda Xc: sig_nmse(Xc, mte)
            else:
                summ["metric"] = "fault_band_ratio"
                summ["fb_y"] = float(fault_band_energy_ratio(Yte, mte["f_fault"]).mean()); summ["fb_xhat"] = float(fault_band_energy_ratio(xhat_te, mte["f_fault"]).mean())
                summ["kurt_y"] = float(kurtosis(Yte).mean()); summ["kurt_xhat"] = float(kurtosis(xhat_te).mean())
                rng0 = np.random.default_rng(0); xr = np.zeros_like(xhat_te)
                for i in range(len(xr)):
                    idx = rng0.choice(xr.shape[1], max(1, int(np.sum(xhat_te[i] != 0))), replace=False); xr[i, idx] = rng0.standard_normal(len(idx))
                summ["fb_random"] = float(fault_band_energy_ratio(xr, mte["f_fault"]).mean()); summ["f_res"] = mte["f_res"]; summ["f_fault"] = mte["f_fault"]
                met = lambda Xc: float(fault_band_energy_ratio(Xc, mte["f_fault"]).mean())

            starts = {"cold": None}; summ["heads"] = {}
            heads_here = [h for h in args.heads if not (kind != "cwru" and h == "conv_lista")]
            for kind_h in heads_here:
                t0 = time.perf_counter()
                head = make_head(kind_h, Ad, args.K, mu=1.0 / A.sigma_max() ** 2)
                f = out / "heads" / f"{kind_h}_K{args.K}_xhat.pt"
                fp = dict(atom=[round(float(v), 12) for v in np.asarray(Ad[:, 0])[:40]], shape=list(Ad.shape), n_train=int(Ytr.shape[0]),
                          steps=args.steps, K=args.K, kind=kind_h, seed=seed)
                side, lock = f.with_suffix(".json"), f.with_suffix(".lock")
                while lock.exists():
                    time.sleep(30)
                fresh = f.exists() and side.exists() and json.load(open(side)).get("fingerprint") == fp
                trained_now = not ((args.skip_train and f.exists()) or fresh)
                if not trained_now:
                    head.load_state_dict(torch.load(f))
                    train_s = json.load(open(side)).get("train_time_s") if side.exists() else None
                else:
                    lock.write_text("training")
                    try:
                        train_head(head, Ytr, xhat_tr, mtr["lam"], dict(batch=64, lr=5e-4, steps_per_stage=args.steps, layerwise=True, loss="last"),
                                   seed=seed, log=lambda s: None)
                        torch.save(head.state_dict(), f)
                        train_s = time.perf_counter() - t0
                        json.dump(dict(fingerprint=fp, train_time_s=train_s), open(side, "w"))
                    finally:
                        lock.unlink(missing_ok=True)
                x0 = head_outputs(head, Yte, lam); starts[kind_h] = x0
                x_tr = head_outputs(head, Ytr, mtr["lam"])
                d_tr = np.linalg.norm(x_tr - xhat_tr, axis=1) / np.linalg.norm(xhat_tr, axis=1)
                summ["heads"][kind_h] = dict(train_time_s=train_s if train_s is not None else time.perf_counter() - t0, trained_now=trained_now,
                                             n_params=int(sum(p.numel() for p in head.parameters())),
                                             d_rel_train_median=float(np.median(d_tr)), nmse_vs_xhat_db=nmse_db(x0, xhat_te),
                                             d_rel_median=float(np.median(np.linalg.norm(x0 - xhat_te, axis=1) / xn)), metric=met(x0))
                print(f"  head {kind_h}: d_rel {summ['heads'][kind_h]['d_rel_median']:.3f}  metric {summ['heads'][kind_h]['metric']:.3f}  [{time.perf_counter()-t0:.0f}s]")

            t0 = time.perf_counter(); R = {}; viol = 0; checks = 0; ex = None
            for meth in ("ista_banach", "fista_gap"):
                R[meth] = {}
                for sname, x0 in starts.items():
                    R[meth][sname] = {}
                    d = np.linalg.norm((0 if x0 is None else x0) - xhat_te, axis=1)
                    for tau in TAUS:
                        r = (run_ista_certified(A, Yte, lam, eps, mu, x0, tau, xhat=xhat_te, mode="banach", rho=rho) if meth == "ista_banach"
                             else run_fista_gap(A, Yte, lam, eps, mu, x0, tau, xhat=xhat_te))
                        ent = dict(k_star_mean=float(r.k_star.mean()), k_star_std=float(r.k_star.std()), k_star_p90=float(np.percentile(r.k_star, 90)),
                                   metric=met(r.x), time_per_window_ms=1e3 * r.time_s / Yte.shape[0], tightness=percentiles(r.ratios[0]) if r.ratios else None,
                                   rel_err_max=float(np.max(np.linalg.norm(r.x - xhat_te, axis=1) / xn)))
                        if meth == "ista_banach":
                            b = theorem2_bound(rho, d, tau * xn / (1 + 2 * tau)); ent["bound_mean"] = float(b.mean()); ent["bound_dominates"] = int(np.sum(b >= r.k_star))
                        R[meth][sname][str(tau)] = ent; viol += r.violations; checks += r.checks
                        if meth == "ista_banach" and sname == "lista_cp" and tau == 1e-3:
                            ex = dict(k_star=int(r.k_star[0]), x_out=r.x[0], y=Yte[0], xhat=xhat_te[0], x0=x0[0])
                    print(f"  {meth:12s} {sname:9s} k* " + " ".join(f"{t:g}:{R[meth][sname][str(t)]['k_star_mean']:6.1f}" for t in TAUS) + f"  metric(1e-3) {R[meth][sname]['0.001']['metric']:.3f}")
            summ["results"] = R; summ["E1"] = dict(violations=viol, checks=checks); summ["timing"]["certified_s"] = time.perf_counter() - t0
            summ["timing"]["total_s"] = time.perf_counter() - T0
            with open(out / "summary.json", "w") as fh:
                json.dump(summ, fh, indent=1, default=float)
            if ex is not None:
                if kind == "ecg":
                    ex.update(s=mte["S"][0], noisy=mte["S"][0] + mte["N"][0], recon=ex["x_out"] @ mte["Psi"].T, fs=mte["fs"], record=str(mte["record_ids"][0]))
                else:
                    ex.update(fs=mte["fs"], f_fault=mte["f_fault"], atom=mte["atom"])
                np.savez_compressed(out / "example.npz", **ex)
            print(f"  E1 {viol}/{checks}; total {summ['timing']['total_s']:.0f}s")


if __name__ == "__main__":
    main()
