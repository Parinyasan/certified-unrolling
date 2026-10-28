from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cu.algos.certificates import GAP_FLOOR_C
from cu.algos.fp_bounds import (U32, U64, abs_norm_bound, cheap_sigma_bounds, gap_eval_bound, kw_lanczos_inflation,
                                lanczos_estimate, power_estimate, step_error_bound, verified_sigma)
from cu.algos.prox import elastic_net_prox
from cu.algos.tail import ContractiveTail
from cu.eval.experiments.m6_endtoend import load_problem

TAUS = [1e-2, 1e-3, 1e-4]
EPS_MACH = float(np.finfo(np.float64).eps)


def contraction(mu, smin2, smax2, eps):
    LG = max(abs(1 - mu * smin2), abs(mu * smax2 - 1))
    return LG / (1 + mu * eps)


def spectral(name, seed, A, eps, rng):
    Ad = A.to_dense()
    m, n = Ad.shape
    t0 = time.perf_counter()
    ver = verified_sigma(Ad, lower=(m >= n))
    smax = ver["svd_sigma_max"]
    smin2 = ver["svd_sigma_min_gram"] ** 2 if m >= n else 0.0
    mu = 2.0 / (smin2 + smax ** 2)
    rho = contraction(mu, smin2, smax ** 2, eps)
    fac = rho / (1 - rho)
    out = dict(dataset=name, seed=seed, m=m, n=n, eps=eps, eps_rel=eps / smax ** 2, sigma_max=smax, sigma_min_gram=math.sqrt(smin2),
               rho=rho, factor=fac, mu_smax2=mu * smax ** 2, mu_contractive_max_smax2=2.0 / (1 - eps / smax ** 2) if smin2 == 0 else None,
               verified=dict(sigma_bar=ver["sigma_bar"], excess_rel=ver["sigma_bar"] / smax - 1, sigma_low=ver["sigma_low"],
                             low_deficit_rel=(1 - ver["sigma_low"] / math.sqrt(smin2)) if (ver["sigma_low"] and smin2 > 0) else None))
    sl2 = (ver["sigma_low"] or 0.0) ** 2
    mu_v = 2.0 / (sl2 + ver["sigma_bar"] ** 2)
    rho_v = max(abs(1 - mu_v * sl2), abs(mu_v * ver["sigma_bar"] ** 2 - 1)) / (1 + mu_v * eps)
    out["verified"].update(rho=rho_v, factor=rho_v / (1 - rho_v), d_rho=rho_v - rho, d_factor=rho_v / (1 - rho_v) - fac)
    out["cheap"] = {}
    for k, sb in cheap_sigma_bounds(Ad).items():
        mu_c = 2.0 / sb ** 2
        r_c = 1.0 / (1 + mu_c * eps)
        out["cheap"][k] = dict(sigma=sb, sigma2_ratio=sb ** 2 / smax ** 2, factor=r_c / (1 - r_c), factor_ratio=(r_c / (1 - r_c)) / fac)
    out["power"] = {}
    for k in (20, 30, 100):
        est = np.array([power_estimate(Ad, k, rng) for _ in range(20)]) / smax - 1
        out["power"][str(k)] = dict(median=float(np.median(est)), worst=float(est.min()))
    out["lanczos"] = {}
    for k in (20, 50, 100):
        est = np.array([lanczos_estimate(Ad, k, rng) for _ in range(5)])
        c = kw_lanczos_inflation(n, k, 1e-10)
        sb = float(np.median(est)) * math.sqrt(c) if math.isfinite(c) else float("inf")
        ent = dict(median=float(np.median(est) / smax - 1), kw_sigma2_inflation=c)
        if math.isfinite(sb):
            mu_k = 2.0 / sb ** 2
            r_k = 1.0 / (1 + mu_k * eps)
            ent.update(kw_sigma2_ratio=sb ** 2 / smax ** 2, kw_factor=r_k / (1 - r_k), kw_rho=r_k)
        out["lanczos"][str(k)] = ent
    out["underestimate"] = {}
    for dfc in (1e-3, 5e-3, 1e-2, -out["power"]["30"]["worst"]):
        sh = smax * (1 - dfc)
        mu_u = 2.0 / sh ** 2
        claimed = 1.0 / (1 + mu_u * eps)
        true = contraction(mu_u, smin2, smax ** 2, eps)
        out["underestimate"][f"{dfc:.4g}"] = dict(claimed_rho=claimed, true_rho=true, contractive=bool(true < 1),
                                                  claimed_factor=claimed / (1 - claimed), true_factor=(true / (1 - true)) if true < 1 else None)
    out["time_s"] = time.perf_counter() - t0
    return out


def float32_ablation(Ad, Y, lam, eps, xhat, sbar, max_iter=3000):
    m, n = Ad.shape
    absA = np.abs(Ad)
    B = Y.shape[0]
    mu = 2.0 / sbar ** 2
    rho = (1.0 / (1 + mu * eps)) * (1 + 4 * U64)
    fac = rho / (1 - rho)
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    A32 = Ad.astype(np.float32); Y32 = Y.astype(np.float32); lam32 = lamc.astype(np.float32)
    mu32 = np.float32(mu); eps32 = np.float32(eps)
    x = np.zeros((B, n), np.float32)
    V = ("plain", "measured", "rigorous")
    kst = {v: {t: np.full(B, -1) for t in TAUS} for v in V}
    viol = {v: {t: 0 for t in TAUS} for v in V}
    checks = {v: {t: 0 for t in TAUS} for v in V}
    f64 = {t: np.zeros(B, int) for t in TAUS}
    rec = dict(e_rel=[], b64_rel=[], guard_rel=[])
    for k in range(1, max_iter + 1):
        v = x - mu32 * ((x @ A32.T - Y32) @ A32)
        xn = (np.sign(v) * np.maximum(np.abs(v) - mu32 * lam32, 0)) / (1 + mu32 * eps32)
        d32 = np.linalg.norm(xn - x, axis=1).astype(np.float64)
        x64 = x.astype(np.float64); xn64 = xn.astype(np.float64)
        T64 = elastic_net_prox(x64 - mu * ((x64 @ Ad.T - Y) @ Ad), mu * lamc, mu * eps)
        e = np.linalg.norm(xn64 - T64, axis=1)
        b64, _ = step_error_bound(Ad, absA, Y, x64, lam, eps, mu, U64)
        nx = np.linalg.norm(xn64, axis=1)
        nx_low = nx * (1 - (n + 2) * U64)
        err = np.linalg.norm(xn64 - xhat, axis=1)
        C = dict(plain=fac * d32,
                 measured=(rho * d32 * (1 + (n + 3) * U32) + e) / (1 - rho),
                 rigorous=(rho * d32 * (1 + (n + 3) * U32) + e * (1 + (n + 3) * U64) + b64) / (1 - rho) * (1 + 8 * U64))
        cheap = (rho * d32 / (1 - rho)) <= np.array(TAUS)[:, None] * (nx_low - rho * d32 / (1 - rho))
        if k <= 300:
            rec["e_rel"].append(float(np.median(e / nx))); rec["b64_rel"].append(float(np.median(b64 / nx)))
            rec["guard_rel"].append(float(np.median((e * (1 + (n + 3) * U64) + b64) / ((1 - rho) * nx))))
        for v_ in V:
            for i, t in enumerate(TAUS):
                act = kst[v_][t] < 0
                if not act.any():
                    continue
                viol[v_][t] += int(np.sum((C[v_] < err) & act)); checks[v_][t] += int(act.sum())
                if v_ == "rigorous":
                    f64[t] += (act & cheap[i]).astype(int)
                fired = act & (C[v_] <= t * (nx_low - C[v_]))
                kst[v_][t][fired] = k
        x = xn
        if all((kst[v_][t] > 0).all() for v_ in V for t in TAUS):
            break
    res = {}
    for t in TAUS:
        res[str(t)] = {v_: dict(k_mean=float(np.where(kst[v_][t] > 0, kst[v_][t], max_iter).mean()), not_fired=int(np.sum(kst[v_][t] < 0)),
                                violations=viol[v_][t], checks=checks[v_][t]) for v_ in V}
        res[str(t)]["rigorous"]["f64_checks_mean"] = float(f64[t].mean())
        res[str(t)]["rigorous"]["changed_vs_measured"] = int(np.sum(kst["rigorous"][t] != kst["measured"][t]))
    res["median_over_iterations"] = {k_: float(np.median(v_)) for k_, v_ in rec.items()}
    res["rho_certified"] = rho
    return res


def gap_part(Ad, Y, lam, eps, mu, xhat, N_A, max_iter=3000):
    absA = np.abs(Ad)
    m, n = Ad.shape
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    P, D, E, En = gap_eval_bound(Ad, absA, Y, xhat, lam, eps, U64, N_A)
    xn = np.linalg.norm(xhat, axis=1)
    floor_code = GAP_FLOOR_C * EPS_MACH * (np.abs(P) + np.abs(D))
    at = dict(P_minus_D_median=float(np.median(P - D)), floor_code_median=float(np.median(floor_code)),
              E_median=float(np.median(E)), En_median=float(np.median(En)),
              E_over_floor_median=float(np.median(E / floor_code)), En_over_floor_median=float(np.median(En / floor_code)),
              c_equiv_median=float(np.median(E / (EPS_MACH * (np.abs(P) + np.abs(D))))), c_equiv_max=float(np.max(E / (EPS_MACH * (np.abs(P) + np.abs(D))))),
              cert_floor_rel_code=float(np.median(np.sqrt(2 * floor_code / eps) / xn)),
              cert_floor_rel_E=float(np.median(np.sqrt(2 * E / eps) / xn)), cert_floor_rel_En=float(np.median(np.sqrt(2 * En / eps) / xn)))

    def run(kind):
        B = Y.shape[0]
        ks = {v: {t: np.full(B, -1) for t in TAUS} for v in ("code", "rig", "rign")}
        x = np.zeros((B, n)); z = x.copy(); tt = np.ones(B)
        mu_f = 1.0 / float(np.linalg.norm(Ad, 2)) ** 2
        for k in range(1, max_iter + 1):
            if kind == "ista":
                x = elastic_net_prox(x - mu * ((x @ Ad.T - Y) @ Ad), mu * lamc, mu * eps)
            else:
                gz = (z @ Ad.T - Y) @ Ad
                xnew = elastic_net_prox(z - mu_f * gz, mu_f * lamc, mu_f * eps)
                tn = 0.5 * (1 + np.sqrt(1 + 4 * tt * tt)); beta = ((tt - 1) / tn)[:, None]
                back = np.sum((z - xnew) * (xnew - x), 1) > 0
                beta[back] = 0.0; tn[back] = 1.0
                z = xnew + beta * (xnew - x); x = xnew; tt = tn
            P_, D_, E_, En_ = gap_eval_bound(Ad, absA, Y, x, lam, eps, U64, N_A)
            Cc = np.sqrt(2 * np.maximum(P_ - D_, GAP_FLOOR_C * EPS_MACH * (np.abs(P_) + np.abs(D_))) / eps)
            Cr = np.sqrt(2 * np.maximum(P_ - D_ + E_, 0) / eps) * (1 + 4 * U64)
            Crn = np.sqrt(2 * np.maximum(P_ - D_ + En_, 0) / eps) * (1 + 4 * U64)
            nx = np.linalg.norm(x, axis=1); nxl = nx * (1 - (n + 2) * U64)
            for v, C, nn in (("code", Cc, nx), ("rig", Cr, nxl), ("rign", Crn, nxl)):
                for t in TAUS:
                    f = (ks[v][t] < 0) & (C <= t * (nn - C)); ks[v][t][f] = k
            if all((ks[v][t] > 0).all() for v in ks for t in TAUS):
                break
        return {str(t): dict(code=float(ks["code"][t].mean()), rigorous=float(ks["rig"][t].mean()), rigorous_normwise=float(ks["rign"][t].mean()),
                             changed=int(np.sum(ks["code"][t] != ks["rig"][t])), changed_normwise=int(np.sum(ks["code"][t] != ks["rign"][t])),
                             not_fired_rigorous=int(np.sum(ks["rig"][t] < 0))) for t in TAUS}
    return dict(at_xhat=at, ista_gap=run("ista"), fista_gap_fresh=run("fista"))


def float64_banach(Ad, Y, lam, eps, xhat, sbar, max_iter=5000):
    absA = np.abs(Ad)
    B, n = Y.shape[0], Ad.shape[1]
    mu = 2.0 / sbar ** 2
    rho = (1.0 / (1 + mu * eps)) * (1 + 4 * U64)
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    x = np.zeros((B, n))
    ks = {v: {t: np.full(B, -1) for t in TAUS} for v in ("plain", "rigorous")}
    viol = {v: 0 for v in ks}; checks = 0; term = []
    for k in range(1, max_iter + 1):
        b64, _ = step_error_bound(Ad, absA, Y, x, lam, eps, mu, U64)
        xn = elastic_net_prox(x - mu * ((x @ Ad.T - Y) @ Ad), mu * lamc, mu * eps)
        d = np.linalg.norm(xn - x, axis=1); nx = np.linalg.norm(xn, axis=1); err = np.linalg.norm(xn - xhat, axis=1)
        C = dict(plain=rho / (1 - rho) * d, rigorous=(rho * d * (1 + (n + 3) * U64) + b64) / (1 - rho) * (1 + 8 * U64))
        act_any = (ks["rigorous"][TAUS[-1]] < 0)
        checks += int(act_any.sum())
        if k <= 300:
            term.append(float(np.median(b64 / ((1 - rho) * np.maximum(nx, 1e-300)))))
        for v in ks:
            viol[v] += int(np.sum((C[v] < err) & act_any))
            for t in TAUS:
                f = (ks[v][t] < 0) & (C[v] <= t * (nx * (1 - (n + 2) * U64) - C[v])); ks[v][t][f] = k
        x = xn
        if all((ks[v][t] > 0).all() for v in ks for t in TAUS):
            break
    return {str(t): dict(plain=float(ks["plain"][t].mean()), rigorous=float(ks["rigorous"][t].mean()),
                         changed=int(np.sum(ks["plain"][t] != ks["rigorous"][t]))) for t in TAUS} | dict(
        guard_term_rel_median=float(np.median(term)), violations=viol, checks=checks)


def margins(Ad, Y, lam, eps, mu, rho, xhat, max_iter=5000):
    n = Ad.shape[1]
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    x = np.zeros_like(xhat); done = np.zeros(Y.shape[0], bool)
    mb, mg, rb, rg = np.inf, np.inf, np.inf, np.inf
    for k in range(1, max_iter + 1):
        r = x @ Ad.T - Y; g = r @ Ad
        P = 0.5 * np.sum(r * r, 1) + np.sum(lamc * np.abs(x), 1) + 0.5 * eps * np.sum(x * x, 1)
        D = -np.sum(r * Y, 1) - 0.5 * np.sum(r * r, 1) - np.sum(np.maximum(np.abs(g) - lamc, 0) ** 2, 1) / (2 * eps)
        G = np.sqrt(2 * np.maximum(P - D, GAP_FLOOR_C * EPS_MACH * (np.abs(P) + np.abs(D))) / eps)
        e_prev = np.linalg.norm(x - xhat, axis=1)
        xn = elastic_net_prox(x - mu * g, mu * lamc, mu * eps)
        C = rho / (1 - rho) * np.linalg.norm(xn - x, axis=1); err = np.linalg.norm(xn - xhat, axis=1)
        a = ~done
        mb = min(mb, float(np.min((C - err)[a]))); rb = min(rb, float(np.min((C / np.maximum(err, 1e-300))[a])))
        mg = min(mg, float(np.min((G - e_prev)[a]))); rg = min(rg, float(np.min((G / np.maximum(e_prev, 1e-300))[a])))
        done |= C <= 1e-4 * (np.linalg.norm(xn, axis=1) - C)
        x = xn
        if done.all():
            break
    return dict(banach_min_margin=mb, banach_min_ratio=rb, gap_min_margin=mg, gap_min_ratio=rg)


def reference_bound(Ad, Y, lam, eps, mu, rho, xhat):
    absA = np.abs(Ad)
    n = Ad.shape[1]
    lamc = np.asarray(lam, np.float64).reshape(-1, 1)
    T = elastic_net_prox(xhat - mu * ((xhat @ Ad.T - Y) @ Ad), mu * lamc, mu * eps)
    b64, _ = step_error_bound(Ad, absA, Y, xhat, lam, eps, mu, U64)
    bound = (np.linalg.norm(xhat - T, axis=1) * (1 + (n + 3) * U64) + b64) / (1 - rho) * (1 + 8 * U64)
    xn = np.linalg.norm(xhat, axis=1)
    return dict(abs_median=float(np.median(bound)), abs_max=float(bound.max()), rel_median=float(np.median(bound / xn)),
                frac_above_1e9=float(np.mean(bound > 1e-9)), b64_rel_median=float(np.median(b64 / xn)),
                exact_arith_claim=float(rho / (1 - rho) * 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["spectral", "float32", "gap", "reference"])
    ap.add_argument("--synthetic-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--real", nargs="*", default=["ecg_compressive", "ecg_denoise_overcomplete", "cwru_or007", "cwru_ir007"])
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--out", default="results/m9/numerics.json")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    S = json.load(open(out)) if out.exists() else {}
    rng = np.random.default_rng(20261022)
    problems = [("synthetic", s) for s in args.synthetic_seeds] + [(r, 0) for r in args.real]
    for name, seed in problems:
        T0 = time.perf_counter()
        A, Y, lam, eps, _ = load_problem(name, seed, args.n_test if name == "synthetic" else 500)
        Ad = A.to_dense()
        key = f"{name}/seed{seed}"
        ent = S.get(key, {})
        tail = ContractiveTail(A, lam, eps)
        mu, rho = tail.mu, tail.rho()
        xhat, _, fd = tail.solve_fixed_point(Y, tol=1e-12)
        if "spectral" in args.parts:
            ent["spectral"] = spectral(name, seed, A, eps, rng)
            sp = ent["spectral"]
            print(f"{key}: sigma_max {sp['sigma_max']:.6f} verified excess {sp['verified']['excess_rel']:.2e} rho {sp['rho']:.5f} "
                  f"power30 worst {sp['power']['30']['worst']:+.2e} median {sp['power']['30']['median']:+.2e} | "
                  f"aat_inf factor x{sp['cheap']['aat_inf']['factor_ratio']:.2f} | Lanczos100+KW factor x{sp['lanczos']['100'].get('kw_factor', float('nan')) / sp['factor']:.4f}", flush=True)
        if "reference" in args.parts:
            ent["reference"] = reference_bound(Ad, Y, lam, eps, mu, rho, xhat)
            print(f"{key}: reference bound median {ent['reference']['abs_median']:.2e} max {ent['reference']['abs_max']:.2e} (frac > 1e-9: {ent['reference']['frac_above_1e9']:.3f})", flush=True)
        if "margins" in args.parts:
            ent["margins"] = margins(Ad, Y, lam, eps, mu, rho, xhat)
            print(f"{key}: min margin Banach {ent['margins']['banach_min_margin']:.2e} (ratio {ent['margins']['banach_min_ratio']:.3f}), "
                  f"gap {ent['margins']['gap_min_margin']:.2e} (ratio {ent['margins']['gap_min_ratio']:.3f})", flush=True)
        if name == "synthetic" and "float64" in args.parts:
            sbar = ent.get("spectral", {}).get("verified", {}).get("sigma_bar") or verified_sigma(Ad)["sigma_bar"]
            ent["float64"] = float64_banach(Ad, Y, lam, eps, xhat, sbar)
            print(f"{key} float64 Banach: " + " | ".join(f"tau {t:g}: {ent['float64'][str(t)]['plain']:.2f} -> {ent['float64'][str(t)]['rigorous']:.2f} "
                  f"(changed {ent['float64'][str(t)]['changed']})" for t in TAUS) + f"; term {ent['float64']['guard_term_rel_median']:.1e}, viol {ent['float64']['violations']}", flush=True)
        if name == "synthetic" and "float32" in args.parts:
            sbar = ent.get("spectral", {}).get("verified", {}).get("sigma_bar") or verified_sigma(Ad)["sigma_bar"]
            ent["float32"] = float32_ablation(Ad, Y, lam, eps, xhat, sbar)
            for t in TAUS:
                r = ent["float32"][str(t)]
                print(f"{key} float32 tau {t:g}: plain k {r['plain']['k_mean']:.2f} viol {r['plain']['violations']} | measured k {r['measured']['k_mean']:.2f} viol {r['measured']['violations']} | "
                      f"rigorous k {r['rigorous']['k_mean']:.2f} viol {r['rigorous']['violations']} not fired {r['rigorous']['not_fired']} f64 checks {r['rigorous']['f64_checks_mean']:.2f} changed {r['rigorous']['changed_vs_measured']}", flush=True)
        if name == "synthetic" and "gap" in args.parts:
            N_A = abs_norm_bound(np.abs(Ad))
            ent["gap"] = gap_part(Ad, Y, lam, eps, mu, xhat, N_A)
            g = ent["gap"]
            print(f"{key} gap at xhat: floor code {g['at_xhat']['cert_floor_rel_code']:.2e} rigorous {g['at_xhat']['cert_floor_rel_E']:.2e} normwise {g['at_xhat']['cert_floor_rel_En']:.2e} (c equiv {g['at_xhat']['c_equiv_median']:.0f})", flush=True)
            for t in TAUS:
                a, b = g["ista_gap"][str(t)], g["fista_gap_fresh"][str(t)]
                print(f"   tau {t:g}: ISTA+gap {a['code']:.2f} -> {a['rigorous']:.2f} (changed {a['changed']}), normwise {a['rigorous_normwise']:.2f} | "
                      f"FISTA+gap {b['code']:.2f} -> {b['rigorous']:.2f} (changed {b['changed']}), normwise {b['rigorous_normwise']:.2f}", flush=True)
        ent["time_s"] = time.perf_counter() - T0
        S[key] = ent
        with open(out, "w") as f:
            json.dump(S, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
