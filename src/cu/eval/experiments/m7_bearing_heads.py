from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cu.algos.certified_solve import fista_gap, head_cast, head_forward, head_params, ista_banach
from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.data.cwru import load_cwru
from cu.eval.experiments.m6_endtoend import cold_error_trace
from cu.models.lista import make_head
from cu.train.train_head import head_outputs, train_head

FILES = {"cwru_or007": "130", "cwru_ir007": "105"}
TAUS = [1e-2, 1e-3, 1e-4]


def rel(x, xhat):
    return np.linalg.norm(x - xhat, axis=1) / np.linalg.norm(xhat, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=list(FILES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dense-hop", type=int, default=16)
    ap.add_argument("--out", default="results/m7")
    ap.add_argument("--train-dense-only", action="store_true", help="only train and cache the dense-windowing heads")
    args = ap.parse_args()

    for name in args.files:
        base = dict(file=FILES[name], n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2)
        for seed in args.seeds:
            T0 = time.perf_counter()
            out = Path(args.out) / name; (out / f"seed{seed}" / "heads").mkdir(parents=True, exist_ok=True)
            if args.train_dense_only:
                A, _, _, _ = load_cwru(base, 5, seed, "train")
                _, Yd, _, md = load_cwru(dict(base, hop=args.dense_hop), 6400, seed, "train")
                Ad = A.to_dense(); smax2 = A.sigma_max() ** 2
                tail = ContractiveTail(A, md["lam"], md["eps"])
                xh_d, _ = compute_and_cache_xhat(Yd, A, md["lam"], md["eps"], mu=tail.mu, tol=1e-10, max_iter=200_000)
                for kind in ("lista_cp", "conv_lista"):
                    f = out / f"seed{seed}" / "heads" / f"{kind}_dense_K{args.K}.pt"
                    if f.exists():
                        continue
                    t0 = time.perf_counter(); head = make_head(kind, Ad, args.K, mu=1.0 / smax2)
                    train_head(head, Yd, xh_d, md["lam"], dict(batch=64, lr=5e-4, steps_per_stage=args.steps, layerwise=True, loss="last"), seed=seed, log=lambda s: None)
                    torch.save(head.state_dict(), f)
                    print(f"{name} seed {seed}: trained {kind}_dense on {Yd.shape[0]} windows [{time.perf_counter() - t0:.0f}s]", flush=True)
                continue
            A, Yn, _, mn = load_cwru(base, 6400, seed, "train")
            _, Yd, _, md = load_cwru(dict(base, hop=args.dense_hop), 6400, seed, "train")
            _, Yt, _, mt = load_cwru(base, 500, seed + 1000, "test")
            Ad = A.to_dense(); eps = mt["eps"]; smax2 = A.sigma_max() ** 2
            tail = ContractiveTail(A, mt["lam"], eps); mu = tail.mu; rho = tail.rho()
            xh_t, _ = compute_and_cache_xhat(Yt, A, mt["lam"], eps, mu=mu, tol=1e-12, max_iter=200_000)
            xh_n, _ = compute_and_cache_xhat(Yn, A, mn["lam"], eps, mu=mu, tol=1e-10, max_iter=200_000)
            xh_d, info_d = compute_and_cache_xhat(Yd, A, md["lam"], eps, mu=mu, tol=1e-10, max_iter=200_000)
            S = dict(file=name, seed=seed, K=args.K, steps=args.steps, n_train_native=int(Yn.shape[0]), n_train_dense=int(Yd.shape[0]),
                     n_test=int(Yt.shape[0]), dense_hop=args.dense_hop, rho=rho, xhat_dense_time_s=info_d["time_s"], heads={})
            print(f"=== {name} seed {seed}: train native {Yn.shape[0]}, dense {Yd.shape[0]}, test {Yt.shape[0]}", flush=True)

            cold = {}
            for tname in ("ista_banach", "fista_gap"):
                cold[tname] = {}
                for tau in TAUS:
                    _, k = (ista_banach(Ad, Yt, mt["lam"], eps, mu, rho, None, tau) if tname == "ista_banach"
                            else fista_gap(Ad, Yt, mt["lam"], eps, 1.0 / smax2, None, tau))
                    cold[tname][str(tau)] = k
            traces = {t: cold_error_trace(Ad, Yt, mt["lam"], eps, mu, xh_t, t, 300, 1.0 / smax2) for t in ("ista_banach", "fista_gap")}
            S["cold"] = {t: {tau: float(v.mean()) for tau, v in d.items()} for t, d in cold.items()}

            for kind in ("lista_cp", "conv_lista"):
                for data in ("native", "dense"):
                    key = f"{kind}_{data}"
                    t0 = time.perf_counter()
                    head = make_head(kind, Ad, args.K, mu=1.0 / smax2)
                    reuse = Path("results/m5r/e6") / name / f"seed{seed}" / "heads" / f"{kind}_K{args.K}_xhat.pt"
                    f = out / f"seed{seed}" / "heads" / f"{key}_K{args.K}.pt"
                    if data == "native" and reuse.exists():
                        head.load_state_dict(torch.load(reuse)); trained = "reused (results/m5r/e6)"
                    elif f.exists():
                        head.load_state_dict(torch.load(f)); trained = "cached"
                    else:
                        Ytr, Xtr, ltr = (Yn, xh_n, mn["lam"]) if data == "native" else (Yd, xh_d, md["lam"])
                        train_head(head, Ytr, Xtr, ltr, dict(batch=64, lr=5e-4, steps_per_stage=args.steps, layerwise=True, loss="last"),
                                   seed=seed, log=lambda s: None)
                        torch.save(head.state_dict(), f); trained = "trained"
                    t_train = time.perf_counter() - t0
                    x_tr = head_outputs(head, Yn, mn["lam"]); x0 = head_outputs(head, Yt, mt["lam"])
                    per_layer = [float(np.median(rel(head_outputs(head, Yt, mt["lam"], n_layers=k + 1), xh_t))) for k in range(args.K)]
                    d_t = rel(x0, xh_t)
                    ent = dict(trained=trained, train_s=t_train, n_params=int(sum(p.numel() for p in head.parameters())),
                               d_rel_test_median=float(np.median(d_t)), d_rel_test_mean=float(d_t.mean()),
                               d_rel_train_median=float(np.median(rel(x_tr, xh_n))), per_layer_d_rel_median=per_layer, tails={})
                    for tname in ("ista_banach", "fista_gap"):
                        E = traces[tname]; reach = E <= d_t[:, None]
                        keq = np.where(reach.any(1), reach.argmax(1), E.shape[1])
                        ent["tails"][tname] = dict(k_eq_mean=float(keq.mean()))
                        for tau in TAUS:
                            xo, k = (ista_banach(Ad, Yt, mt["lam"], eps, mu, rho, x0, tau) if tname == "ista_banach"
                                     else fista_gap(Ad, Yt, mt["lam"], eps, 1.0 / smax2, x0, tau))
                            assert np.all(rel(xo, xh_t) <= tau * (1 + 1e-9))
                            kc = cold[tname][str(tau)]
                            ent["tails"][tname][str(tau)] = dict(k_mean=float(k.mean()), cold_mean=float(kc.mean()),
                                                                 total_pairs=float(args.K + k.mean()), net_factor_pairs=float(kc.mean() / (args.K + k.mean())),
                                                                 saved_mean=float((kc - k).mean()))
                    S["heads"][key] = ent
                    e3 = ent["tails"]["ista_banach"]["0.001"]; f3 = ent["tails"]["fista_gap"]["0.001"]
                    print(f"{key:20s} [{trained}] params {ent['n_params']:>9d}  d_rel train {ent['d_rel_train_median']:.3f} test {ent['d_rel_test_median']:.3f}  "
                          f"k_eq ISTA {ent['tails']['ista_banach']['k_eq_mean']:.1f}  ISTA {e3['cold_mean']:.1f}->{e3['k_mean']:.1f} (net {e3['net_factor_pairs']:.2f})  "
                          f"FISTA {f3['cold_mean']:.1f}->{f3['k_mean']:.1f} (net {f3['net_factor_pairs']:.2f})  [{t_train:.0f}s]", flush=True)
            S["time_s"] = time.perf_counter() - T0
            with open(out / f"seed{seed}.json", "w") as fh:
                json.dump(S, fh, indent=1)
            print(f"wrote {out / f'seed{seed}.json'} ({S['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
