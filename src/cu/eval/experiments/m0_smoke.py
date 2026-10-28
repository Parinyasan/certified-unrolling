from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.ista import ista
from cu.algos.tail import ContractiveTail
from cu.config import config_hash, load_config
from cu.data.synthetic import add_noise, conditioned_operator, gaussian_operator, lam_from_y, sparse_signals
from cu.eval.metrics import certificate_ratio, nmse_db, percentiles, theorem2_bound, worst_case_ratio


def build_problem(cfg: dict, B: int, seed: int):
    p = cfg["problem"]
    rng = np.random.default_rng(seed)
    if p["A"]["type"] == "gaussian":
        A = gaussian_operator(p["m"], p["n"], rng, col_norm=p["A"].get("col_norm", 1.0))
    elif p["A"]["type"] == "conditioned":
        A = conditioned_operator(p["n"], p["A"]["kappa"], rng, m=p["m"])
    else:
        raise ValueError(p["A"]["type"])
    X = sparse_signals(B, A.n, p["sparsity_p"], rng, p.get("amplitude", {}).get("std", 1.0))
    Y = add_noise(A.matvec(X), p.get("snr_db"), rng)
    lam = lam_from_y(A, Y, p["lam"])
    eps = float(p["eps_rel"]) * A.sigma_max() ** 2
    return A, Y, X, lam, eps


def empirical_lipschitz(tail, Y, lam, xhat, n_pairs, rng):
    B, n = xhat.shape
    idx = rng.integers(0, B, size=n_pairs)
    scale = np.linalg.norm(xhat, axis=1)[idx, None] / np.sqrt(n)
    X = xhat[idx] + scale * rng.standard_normal((n_pairs, n))
    Xp = X + scale * rng.uniform(0.01, 3.0, size=(n_pairs, 1)) * rng.standard_normal((n_pairs, n))
    num = np.linalg.norm(tail.step(X, Y[idx], lam[idx]) - tail.step(Xp, Y[idx], lam[idx]), axis=1)
    den = np.linalg.norm(X - Xp, axis=1)
    return float((num / den).max())


def run_config(cfg_path: str, B: int, seed: int, eps_tol_rel: float, K_head: int, out_root: Path):
    cfg = load_config(cfg_path)
    name = Path(cfg_path).stem
    h = config_hash(dict(cfg=cfg, B=B, seed=seed, eps_tol_rel=eps_tol_rel, K_head=K_head))
    out_dir = out_root / h
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}  (hash {h}, B={B}, seed={seed}) ===")
    t_all = time.perf_counter()

    A, Y, X_star, lam, eps = build_problem(cfg, B, seed)
    tail = ContractiveTail(A, lam, eps)
    rho = tail.rho()
    print(f"A {A.shape}: smax={A.sigma_max():.4f} smin={A.sigma_min():.4f} mu={tail.mu:.5f} eps={eps:.4g} rho={rho:.6f}")

    t0 = time.perf_counter()
    xhat, info = compute_and_cache_xhat(Y, A, lam, eps, mu=tail.mu, tol=1e-12, max_iter=int(cfg["tail"]["max_iter"]))
    print(f"xhat: {'cached' if info['cached'] else 'computed'} in {info['time_s']:.2f}s; iters mean {info['n_iter'].mean():.0f} "
          f"max {info['n_iter'].max()}; NMSE(xhat, x*) = {nmse_db(xhat, X_star):.2f} dB  [{time.perf_counter()-t0:.2f}s]")

    rng = np.random.default_rng(seed + 100)
    t0 = time.perf_counter()
    lip = empirical_lipschitz(tail, Y, lam, xhat, 4000, rng)
    print(f"empirical Lipschitz on 4000 pairs: {lip:.6f}  (rho = {rho:.6f}; ok={lip <= rho*(1+1e-10)})  [{time.perf_counter()-t0:.2f}s]")

    xhat_norm = np.linalg.norm(xhat, axis=1)
    eps_tol = np.maximum(eps_tol_rel * xhat_norm, 1e-7 * np.linalg.norm(Y, axis=1))

    res_c = tail.run_until(None, Y, eps_tol, xhat=xhat)
    slack = 10 * tail.certificate_factor() * 1e-12
    viol = int(np.sum(res_c.certs < res_c.errs_vs_xhat - slack))
    n_checks = int(res_c.certs.size)
    r = certificate_ratio(res_c.certs, res_c.errs_vs_xhat)
    mask = np.arange(res_c.certs.shape[1])[None, :] < res_c.k_star[:, None]
    r_pct = percentiles(r[mask])
    bound_c = theorem2_bound(rho, xhat_norm, eps_tol)
    print(f"cold start: k* mean {res_c.k_star.mean():.1f} (median {np.median(res_c.k_star):.0f}, max {res_c.k_star.max()}); "
          f"Theorem-2 bound mean {bound_c.mean():.0f}, dominates {int(np.sum(bound_c >= res_c.k_star))}/{B}; "
          f"tail time {res_c.time_s:.2f}s")
    print(f"certificate checks: {n_checks}, violations: {viol}")
    print(f"tightness r_k (k <= k*): median {r_pct['p50']:.1f}, p90 {r_pct['p90']:.1f}, max {r_pct['p100']:.1f}; "
          f"worst-case (1+rho)/(1-rho) = {worst_case_ratio(rho):.1f}")
    print(f"NMSE(x_out, x*) = {nmse_db(res_c.x, X_star):.2f} dB; max ||x_out - xhat||/eps_tol = "
          f"{(np.linalg.norm(res_c.x - xhat, axis=1) / eps_tol).max():.3f}")

    t0 = time.perf_counter()
    x0_w = ista(A, Y, lam, K_head, mu=1.0 / A.sigma_max() ** 2)
    d_w = np.linalg.norm(x0_w - xhat, axis=1)
    res_w = tail.run_until(x0_w, Y, eps_tol, xhat=xhat)
    viol_w = int(np.sum(res_w.certs < res_w.errs_vs_xhat - slack))
    bound_w = theorem2_bound(rho, d_w, eps_tol)
    print(f"warm start ({K_head} ISTA steps): d/||xhat|| median {np.median(d_w / xhat_norm):.3f}; k* mean {res_w.k_star.mean():.1f}; "
          f"saved {res_c.k_star.mean() - res_w.k_star.mean():.1f} iterations; Theorem-2 bound mean {bound_w.mean():.0f} "
          f"dominates {int(np.sum(bound_w >= res_w.k_star))}/{B}; violations {viol_w}  [{time.perf_counter()-t0:.2f}s]")

    total = time.perf_counter() - t_all
    print(f"wall-clock for {name}: {total:.2f}s")

    summary = dict(
        config=cfg_path, hash=h, B=B, seed=seed, m=A.m, n=A.n, sigma_max=A.sigma_max(), sigma_min=A.sigma_min(),
        mu=tail.mu, eps=eps, rho=rho, worst_case_ratio=worst_case_ratio(rho), eps_tol_rel=eps_tol_rel,
        xhat_iters_mean=float(info["n_iter"].mean()), xhat_iters_max=int(info["n_iter"].max()), xhat_time_s=info["time_s"],
        nmse_xhat_db=nmse_db(xhat, X_star), empirical_lipschitz=lip,
        cold=dict(k_star_mean=float(res_c.k_star.mean()), k_star_median=float(np.median(res_c.k_star)), k_star_max=int(res_c.k_star.max()),
                  bound_mean=float(bound_c.mean()), bound_dominates=int(np.sum(bound_c >= res_c.k_star)), time_s=res_c.time_s,
                  nmse_out_db=nmse_db(res_c.x, X_star)),
        n_checks=n_checks, violations=viol, tightness=r_pct,
        warm_ista=dict(K=K_head, d_rel_median=float(np.median(d_w / xhat_norm)), k_star_mean=float(res_w.k_star.mean()),
                       bound_mean=float(bound_w.mean()), bound_dominates=int(np.sum(bound_w >= res_w.k_star)), violations=viol_w),
        wallclock_s=total,
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out_dir / "cold_trace.npz", certs=res_c.certs, errs=res_c.errs_vs_xhat, k_star=res_c.k_star, rho=rho)
    return summary, res_c, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/synth_base.yaml", "configs/m0_square.yaml"])
    ap.add_argument("--n", type=int, default=200, help="instances per config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps-tol-rel", type=float, default=1e-3)
    ap.add_argument("--K-head", type=int, default=8)
    ap.add_argument("--out", default="results/m0")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    runs = [run_config(c, args.n, args.seed, args.eps_tol_rel, args.K_head, out_root) for c in args.configs]

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), squeeze=False)
    for ax, (summ, res, name) in zip(axes[0], runs):
        certs, errs = res.certs.ravel(), res.errs_vs_xhat.ravel()
        ok = (errs > 0) & (certs > 0)
        rho = summ["rho"]
        lo, hi = errs[ok].min(), certs[ok].max()
        ax.loglog(errs[ok], certs[ok], ".", ms=2, alpha=0.3, label=f"$(\\|x_k-\\hat x\\|, C_k)$, {ok.sum()} pts")
        g = np.array([lo, hi])
        ax.loglog(g, g, "k-", lw=1, label="diagonal $C_k = \\|x_k-\\hat x\\|$")
        ax.loglog(g, worst_case_ratio(rho) * g, "r--", lw=1, label=f"$(1+\\rho)/(1-\\rho)$ = {worst_case_ratio(rho):.0f}")
        ax.set_xlabel("true error $\\|x_k - \\hat x_\\varepsilon\\|_2$")
        ax.set_ylabel("certificate $C_k$")
        ax.set_title(f"{name}: {summ['m']}x{summ['n']}, $\\rho$={rho:.4f}, violations={summ['violations']}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig_path = out_root / "certificate_vs_error.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nfigure: {fig_path}")
    print(f"TOTAL wall-clock: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
