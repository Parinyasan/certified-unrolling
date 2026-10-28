from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("CU_THREADS", "1"))

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certified_solve import (fista_gap, fista_gap_1, head_cast, head_forward, head_forward_1, head_params,
                                     head_stack, ista_banach, ista_banach_1, timed)
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.config import load_config
from cu.data.synthetic import load_synthetic
from cu.models.lista import make_head

TAUS = [1e-2, 1e-3, 1e-4]
REAL = {
    "ecg_compressive": dict(kind="ecg", cfg=dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2),
                            heads=[("lista_cp", 8), ("alista", 8)]),
    "ecg_denoise_overcomplete": dict(kind="ecg", cfg=dict(dictionary="db4+dct", measurement="identity", snr_db=10.0, lam=0.02, eps_rel=1e-2),
                                     heads=[("lista_cp", 8)]),
    "cwru_or007": dict(kind="cwru", cfg=dict(file="130", n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2),
                       heads=[("lista_cp", 8), ("alista", 8), ("conv_lista", 8)]),
    "cwru_ir007": dict(kind="cwru", cfg=dict(file="105", n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2),
                       heads=[("lista_cp", 8), ("alista", 8), ("conv_lista", 8)]),
}


def load_problem(name: str, seed: int, n_test: int):
    if name == "synthetic_acc":
        from cu.eval.experiments.m11_operating_point import problem_cfg
        pc = problem_cfg()
        A, _, _, _ = load_synthetic(pc, 10, seed=seed)
        _, Y, _, m = load_synthetic(pc, n_test, seed=seed + 1000, A=A)
        hd = Path("results/m11") / f"seed{seed}" / "heads"
        return A, Y, m["lam"], m["eps"], {f"K{K}": ("lista_cp", K, hd / f"cp_K{K}_xhat.pt") for K in (8, 16)}
    if name == "synthetic":
        cfg = load_config("configs/synth_base.yaml")
        A, _, _, _ = load_synthetic(cfg["problem"], 10, seed=seed)
        _, Y, _, m = load_synthetic(cfg["problem"], n_test, seed=seed + 1000, A=A)
        hd = Path("results/m3") / f"seed{seed}" / "heads"
        heads = {f"K{K}": ("lista_cp", K, hd / f"cp_K{K}_xhat.pt") for K in (2, 4, 8, 16)}
        return A, Y, m["lam"], m["eps"], heads
    from cu.eval.experiments.m5r_e6 import load
    spec = REAL[name]
    A, _, _, _ = load(spec["kind"], spec["cfg"], 6400, seed, "train")
    _, Y, _, m = load(spec["kind"], spec["cfg"], n_test, seed + 1000, "test")
    hd = Path("results/m5r/e6") / name / f"seed{seed}" / "heads"
    heads = {f"{k}_K{K}": (k, K, hd / f"{k}_K{K}_xhat.pt") for k, K in spec["heads"]}
    return A, Y, m["lam"], m["eps"], heads


def cold_error_trace(Ad, Y, lam, eps, mu, xhat, tail: str, n_iter: int, mu_fista: float):
    from cu.algos.prox import elastic_net_prox
    B, n = Y.shape[0], Ad.shape[1]
    xn = np.linalg.norm(xhat, axis=1)
    E = np.empty((B, n_iter + 1)); x = np.zeros((B, n)); E[:, 0] = 1.0
    th = np.asarray(lam).reshape(-1, 1)
    if tail == "ista_banach":
        for k in range(1, n_iter + 1):
            x = elastic_net_prox(x - mu * ((x @ Ad.T - Y) @ Ad), mu * th, mu * eps)
            E[:, k] = np.linalg.norm(x - xhat, axis=1) / xn
    else:
        z = x.copy(); t = np.ones(B)
        for k in range(1, n_iter + 1):
            gz = (z @ Ad.T - Y) @ Ad
            xnew = elastic_net_prox(z - mu_fista * gz, mu_fista * th, mu_fista * eps)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); beta = ((t - 1) / tn)[:, None]
            back = np.sum((z - xnew) * (xnew - x), 1) > 0
            beta[back] = 0.0; tn[back] = 1.0
            z = xnew + beta * (xnew - x); x = xnew; t = tn
            E[:, k] = np.linalg.norm(x - xhat, axis=1) / xn
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-test", type=int, default=None, help="default: 1000 synthetic, 500 real")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-latency", type=int, default=None, help="inputs for the single-input latency regime (default 200 synthetic, 50 real)")
    ap.add_argument("--out", default="results/m6")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("CU_THREADS", "1")))
    env = dict(threads=os.environ["OMP_NUM_THREADS"], cpu=platform.processor(), numpy=np.__version__, python=platform.python_version())
    try:
        env["blas"] = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
    except Exception:
        pass

    for name in args.datasets:
        for seed in args.seeds:
            n_test = args.n_test or (1000 if name.startswith("synthetic") else 500)
            out = Path(args.out) / name; out.mkdir(parents=True, exist_ok=True)
            T0 = time.perf_counter()
            A, Y, lam, eps, heads = load_problem(name, seed, n_test)
            Ad = np.ascontiguousarray(A.to_dense(), dtype=np.float64)
            smax2 = A.sigma_max() ** 2
            tail = ContractiveTail(A, lam, eps); mu = tail.mu; rho = tail.rho()
            xhat, _ = compute_and_cache_xhat(Y, A, lam, eps, mu=mu, tol=1e-12, max_iter=200_000)
            xn = np.linalg.norm(xhat, axis=1)
            B = Y.shape[0]
            S = dict(dataset=name, seed=seed, B=B, m=int(Ad.shape[0]), n=int(Ad.shape[1]), rho=rho, mu_ista=mu, mu_fista=1.0 / smax2,
                     taus=TAUS, repeats=args.repeats, env=env, starts={})
            starts = {"cold": dict(x0=None, K=0, head_ms=0.0, head64_ms=0.0)}
            for hname, (kind, K, path) in heads.items():
                if not Path(path).exists():
                    print("missing head", path); continue
                h = make_head(kind, Ad, K, mu=1.0 / smax2); h.load_state_dict(torch.load(path))
                p = head_params(h); p32 = head_cast(p, np.float32); p64 = head_cast(p, np.float64)
                x0, t32 = timed(head_forward, p32, Y, lam, repeats=max(5, args.repeats))
                _, t64 = timed(head_forward, p64, Y, lam, repeats=max(5, args.repeats))
                starts[hname] = dict(x0=np.asarray(x0, np.float64), K=K, head_ms=1e3 * t32 / B, head64_ms=1e3 * t64 / B,
                                     stack32=head_stack(p32), stack64=head_stack(p64),
                                     d_rel_median=float(np.median(np.linalg.norm(x0 - xhat, axis=1) / xn)))
            for sname, st in starts.items():
                S["starts"][sname] = dict(K=st["K"], head_ms=st["head_ms"], head64_ms=st["head64_ms"],
                                          d_rel_median=st.get("d_rel_median"), tails={})
            for tname in ("ista_banach", "fista_gap"):
                E = cold_error_trace(Ad, Y, lam, eps, mu, xhat, tname, 1500 if name.endswith("_acc") else 400, 1.0 / smax2)
                for sname, st in starts.items():
                    if st["x0"] is None:
                        continue
                    dh = np.linalg.norm(st["x0"] - xhat, axis=1) / xn
                    reach = E <= dh[:, None]
                    keq = np.where(reach.any(1), reach.argmax(1), E.shape[1])
                    S["starts"][sname].setdefault("k_eq", {})[tname] = dict(mean=float(keq.mean()), median=float(np.median(keq)),
                                                                          frac_censored=float(np.mean(~reach.any(1))))
                    print(f"{name} s{seed} {sname} {tname}: head worth {keq.mean():.1f} cold iterations (K={st['K']})", flush=True)
            for tname in ("ista_banach", "fista_gap"):
                for tau in TAUS:
                    cold = None
                    for sname, st in starts.items():
                        if tname == "ista_banach":
                            (x, k), ts = timed(ista_banach, Ad, Y, lam, eps, mu, rho, st["x0"], tau, repeats=args.repeats)
                        else:
                            (x, k), ts = timed(fista_gap, Ad, Y, lam, eps, 1.0 / smax2, st["x0"], tau, repeats=args.repeats)
                        rel = np.linalg.norm(x - xhat, axis=1) / xn
                        ent = dict(k_mean=float(k.mean()), k_median=float(np.median(k)), k_p90=float(np.percentile(k, 90)), k_max=int(k.max()),
                                   tail_ms=1e3 * ts / B, head_pairs=st["K"], total_pairs=float(st["K"] + k.mean()),
                                   total_ms=st["head_ms"] + 1e3 * ts / B, guarantee=bool(np.all(rel <= tau * (1 + 1e-9))), rel_err_max=float(rel.max()))
                        if sname == "cold":
                            cold = dict(k=k, ent=ent)
                        else:
                            ent["net_factor_pairs"] = cold["ent"]["total_pairs"] / ent["total_pairs"]
                            ent["net_factor_time"] = cold["ent"]["total_ms"] / ent["total_ms"]
                            ent["tail_factor_pairs"] = cold["ent"]["k_mean"] / ent["k_mean"]
                            ent["frac_instances_net_gain"] = float(np.mean(st["K"] + k < cold["k"]))
                            ent["mean_pairs_saved_by_head"] = float(np.mean(cold["k"] - k))
                        S["starts"][sname]["tails"].setdefault(tname, {})[str(tau)] = ent
                    line = " | ".join(f"{s}: {S['starts'][s]['tails'][tname][str(tau)]['total_pairs']:.1f}p {S['starts'][s]['tails'][tname][str(tau)]['total_ms']:.3f}ms"
                                      for s in starts)
                    print(f"{name} s{seed} {tname:11s} tau {tau:g}: {line}", flush=True)
            nl = min(args.n_latency or (200 if name.startswith("synthetic") else 50), B); AT = np.ascontiguousarray(Ad.T); ATY = Y @ Ad; z0 = np.zeros(Ad.shape[1])
            Y32 = Y.astype(np.float32); lam32 = np.asarray(lam, np.float32)
            for sname, st in starts.items():
                lat = dict(n=nl, head_ms=0.0, head64_ms=0.0, tails={})
                if st["x0"] is not None:
                    for key, stack, yy, ll in (("head_ms", st["stack32"], Y32, lam32), ("head64_ms", st["stack64"], Y, np.asarray(lam, np.float64))):
                        def run_head(stack=stack, yy=yy, ll=ll):
                            for i in range(nl):
                                head_forward_1(*stack, ll[i], yy[i])
                        _, th = timed(run_head, repeats=max(5, args.repeats)); lat[key] = 1e3 * th / nl
                for tname in ("ista_banach", "fista_gap"):
                    for tau in TAUS:
                        x0s = [z0 if st["x0"] is None else st["x0"][i] for i in range(nl)]
                        if tname == "ista_banach":
                            def run_tail(tau=tau, x0s=x0s):
                                return [ista_banach_1(Ad, AT, Y[i], float(lam[i]), eps, mu, rho, x0s[i], tau, 100_000)[1] for i in range(nl)]
                        else:
                            def run_tail(tau=tau, x0s=x0s):
                                return [fista_gap_1(Ad, AT, Y[i], ATY[i], float(lam[i]), eps, 1.0 / smax2, x0s[i], tau, 100_000, 50)[1] for i in range(nl)]
                        kk, tt = timed(run_tail, repeats=args.repeats)
                        lat["tails"].setdefault(tname, {})[str(tau)] = dict(k_mean=float(np.mean(kk)), tail_ms=1e3 * tt / nl, total_ms=lat["head_ms"] + 1e3 * tt / nl)
                S["starts"][sname]["latency"] = lat
            for tname in ("ista_banach", "fista_gap"):
                for tau in TAUS:
                    c = S["starts"]["cold"]["latency"]["tails"][tname][str(tau)]["total_ms"]
                    for sname in starts:
                        if sname != "cold":
                            e = S["starts"][sname]["latency"]["tails"][tname][str(tau)]; e["net_factor_time"] = c / e["total_ms"]
                    line = " | ".join(f"{s_}: {S['starts'][s_]['latency']['tails'][tname][str(tau)]['total_ms']:.3f}ms" for s_ in starts)
                    print(f"{name} s{seed} latency {tname:11s} tau {tau:g}: {line}", flush=True)
            S["time_s"] = time.perf_counter() - T0
            with open(out / f"seed{seed}.json", "w") as f:
                json.dump(S, f, indent=1)
            print(f"wrote {out / f'seed{seed}.json'} ({S['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
