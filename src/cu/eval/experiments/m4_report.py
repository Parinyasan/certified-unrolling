from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/m4")
    args = ap.parse_args()
    root = Path(args.root); (root / "figs").mkdir(exist_ok=True, parents=True)
    S = [json.load(open(p / "summary.json")) for p in sorted(root.glob("seed*")) if (p / "summary.json").exists()]
    tols = [str(t) for t in S[0]["eps_tols"]]
    agg = dict(n_seeds=len(S), seeds=[s["seed"] for s in S], rho_B=ms([s["rho_B"] for s in S]), L=ms([s["L"] for s in S]),
               rho_B_bound=ms([s.get("rho_B_bound_formula", s["rho_B"]) for s in S]),
               nmse_readout_xstar_db=ms([s["nmse_readout_xstar_db"] for s in S]), zstar_iters_mean=ms([s["zstar_iters_mean"] for s in S]),
               encoder_d_rel_median=ms([s["encoder_d_rel_median"] for s in S]),
               E1=dict(violations=int(sum(s["E1"]["violations"] for s in S)), checks=int(sum(s["E1"]["checks"] for s in S))),
               tightness_p50=ms([s["E2"]["r"]["p50"] for s in S]), tightness_p90=ms([s["E2"]["r"]["p90"] for s in S]),
               train_time_s=ms([s["timing"]["train_tail_s"] for s in S]), worst_case_ratio=ms([s["worst_case_ratio"] for s in S]))
    agg["E3"] = {st: {t: {k: ms([s["E3"][st][t][k] for s in S]) for k in ("k_star_mean", "bound_mean", "nmse_out_db", "tail_time_s")} for t in tols}
                 for st in ("cold", "warm_encoder")}
    for st in agg["E3"]:
        for t in tols:
            agg["E3"][st][t]["bound_dominates_total"] = int(sum(s["E3"][st][t]["bound_dominates"] for s in S))
            agg["E3"][st][t]["readout_bound_ok_all"] = all(s["E3"][st][t]["readout_err_bound_ok"] for s in S)
    if all("compare" in s for s in S):
        agg["compare"] = {t: {k: ms([s["compare"][t][k] for s in S]) for k in S[0]["compare"][tols[0]]} for t in tols}
        agg["compare"]["rho_A"] = ms([s["compare"]["rho_A"] for s in S]); agg["compare"]["A_nmse_xhat_db"] = ms([s["compare"]["A_nmse_xhat_db"] for s in S])
    with open(root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for s in S:
        h = s["train_hist"]
        axes[0].plot(h["step"], h["val"], lw=1); axes[1].plot(h["step"], h["rho_B"], lw=1); axes[2].plot(h["step"], h["k_fp"], lw=1)
    axes[0].set_ylabel("validation NMSE vs $x^\\star$ (dB)"); axes[1].set_ylabel("$\\rho_B$ (exact, float64)"); axes[2].set_ylabel("fixed-point iterations (train)")
    for ax in axes:
        ax.set_xlabel("JFB training step"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(root / "figs" / "fig_m4_training.png", dpi=160); fig.savefig(root / "figs" / "fig_m4_training.pdf")
    print(json.dumps({k: agg[k] for k in ("rho_B", "nmse_readout_xstar_db", "E1", "tightness_p50")}, indent=1))
    for t in tols:
        print(f"tol {t}: B cold k*={agg['E3']['cold'][t]['k_star_mean']['mean']:.1f} warm k*={agg['E3']['warm_encoder'][t]['k_star_mean']['mean']:.1f} "
              f"NMSE out={agg['E3']['warm_encoder'][t]['nmse_out_db']['mean']:.2f} dB")


if __name__ == "__main__":
    main()
