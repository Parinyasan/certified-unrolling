from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["ista_banach", "ista_banach_mu1", "ista_gap", "ista_rhoS", "fista_gap"]
LABEL = {"ista_banach": "ISTA + Banach ($\\mu=2/\\sigma_{\\max}^2$)", "ista_banach_mu1": "ISTA + Banach ($\\mu=1/\\sigma_{\\max}^2$)",
         "ista_gap": "ISTA + duality gap", "ista_rhoS": "ISTA + certified $\\rho_S$", "fista_gap": "FISTA + duality gap"}
STARTS = ["cold", "cp_K8_xhat", "cp_K16_xhat", "cp_K16_xstar"]
SLAB = {"cold": "cold", "cp_K8_xhat": "K=8", "cp_K16_xhat": "K=16", "cp_K16_xstar": "K=16 ($x^\\star$-trained)"}


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/m3r")
    ap.add_argument("--out", default="results/m3r/figs")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(p for p in root.glob("seed*") if (p / "summary.json").exists())
    S = [json.load(open(p / "summary.json")) for p in seeds]; TR = [np.load(p / "traces.npz") for p in seeds]
    taus = [str(t) for t in S[0]["taus"]]
    agg = dict(n_seeds=len(S), seeds=[s["seed"] for s in S], taus=taus, rho=ms([s["rho"] for s in S]), rho_old=ms([s["rho_old"] for s in S]),
               mu_times_smax2=S[0]["mu_times_smax2"], nmse_xhat_xstar_db=ms([s["nmse_xhat_xstar_db"] for s in S]),
               gap_floor_rel_median=ms([s["gap_floor_rel_median"] for s in S]), cd_check_rel_diff_max=max(s["cd_check_rel_diff_max"] for s in S),
               xhat_time_s=ms([s["xhat_time_s"] for s in S]))
    rho = agg["rho"]["mean"]
    agg["heads"] = {h: {k: ms([s["heads"][h][k] for s in S]) for k in ("d_rel_median", "nmse_vs_xhat_db", "nmse_vs_xstar_db")} for h in S[0]["heads"]}
    agg["E1"] = {m: dict(violations=int(sum(s["E1"][m]["violations"] for s in S)), checks=int(sum(s["E1"][m]["checks"] for s in S))) for m in METHODS}
    agg["E1_total"] = dict(violations=int(sum(agg["E1"][m]["violations"] for m in METHODS)), checks=int(sum(agg["E1"][m]["checks"] for m in METHODS)))
    R = {}
    for m in METHODS:
        R[m] = {}
        for st in S[0]["results"][m]:
            R[m][st] = {}
            for t in taus:
                e = [s["results"][m][st][t] for s in S]
                ent = {k: ms([x[k] for x in e]) for k in ("k_star_mean", "k_star_median", "nmse_out_db", "time_s", "rel_err_max")}
                ent["guarantee_holds_all"] = all(x["guarantee_holds"] for x in e)
                if e[0].get("tightness"):
                    ent["tightness"] = {q: ms([x["tightness"][q] for x in e]) for q in ("p50", "p90", "p99", "p100")}
                if "bound_mean" in e[0]:
                    ent["bound_mean"] = ms([x["bound_mean"] for x in e]); ent["bound_dominates_total"] = int(sum(x["bound_dominates"] for x in e))
                if "frac_S_equals_supp_before_stop" in e[0]:
                    ent["frac_S_equals_supp_before_stop"] = ms([x["frac_S_equals_supp_before_stop"] for x in e])
                ent["speedup_vs_cold"] = ms([s["results"][m]["cold"][t]["k_star_mean"] / s["results"][m][st][t]["k_star_mean"] for s in S])
                ent["speedup_vs_ista_banach_cold"] = ms([s["results"]["ista_banach"]["cold"][t]["k_star_mean"] / s["results"][m][st][t]["k_star_mean"] for s in S])
                R[m][st][t] = ent
    agg["results"] = R
    RS = [json.load(open(p / "rhoS_stats.json")) for p in seeds if (p / "rhoS_stats.json").exists()]
    if RS:
        agg["rhoS_stats"] = dict(n_seeds=len(RS), rho_S_oracle_median=ms([r["rho_S_oracle_median"] for r in RS]),
                                 supp_size_median=ms([r["supp_size_median"] for r in RS]),
                                 taus={t: {k: ms([r["taus"][t][k] for r in RS]) for k in ("rhoS_at_stop_median", "S_over_supp_median", "frac_global_fallback", "S_size_median", "cert_factor_median")}
                                       for t in taus if all(t in r["taus"] for r in RS)})
    agg["E4"] = {k: np.mean([s["E4"][k] for s in S], 0).tolist() for k in ("ista", "fista", "enista", "enfista")}
    agg["per_layer"] = {h: np.mean([s["heads"][h]["per_layer_nmse_xstar"] for s in S], 0).tolist() for h in S[0]["heads"]}
    agg["E5"] = {}
    for e in S[0]["E5"]:
        ent = dict(rho=ms([s["E5"][e]["rho"] for s in S]), nmse_xhat_db=ms([s["E5"][e]["nmse_xhat_db"] for s in S]), rho_S_oracle_median=ms([s["E5"][e]["rho_S_oracle_median"] for s in S]))
        if "k" in S[0]["E5"][e]:
            for key in ("k", "k_warm", "k_fista"):
                ent[key] = {t: ms([s["E5"][e][key][t] for s in S]) for t in taus}
        agg["E5"][e] = ent
    agg["float32"] = {t: dict(no_guard_viol=int(sum(s["float32"][t]["no_guard"]["violations"] for s in S)), guard_viol=int(sum(s["float32"][t]["guard"]["violations"] for s in S)),
                              checks=int(sum(s["float32"][t]["no_guard"]["checks"] for s in S)), guard_k=ms([s["float32"][t]["guard"]["k_star_mean"] for s in S]),
                              no_guard_k=ms([s["float32"][t]["no_guard"]["k_star_mean"] for s in S]),
                              no_guard_not_fired=ms([s["float32"][t]["no_guard"]["frac_not_fired"] for s in S]),
                              guard_not_fired=ms([s["float32"][t]["guard"]["frac_not_fired"] for s in S]),
                              **({"step_error_rel": ms([s["float32"][t]["guard"]["step_error_rel_median"] for s in S]),
                                  "guard_term_rel": ms([s["float32"][t]["guard"]["guard_term_rel_median"] for s in S])}
                                 if "step_error_rel_median" in S[0]["float32"][t]["guard"] else {})) for t in taus}
    if "head_gap" in S[0]:
        agg["head_gap"] = {h: {k: ms([s["head_gap"][h][k] for s in S]) for k in ("d_rel_train_mean", "d_rel_test_mean", "d_rel_train_median", "d_rel_test_median")}
                           for h in S[0]["head_gap"]}
    for s in S:
        if "ideal" in s:
            agg["ideal"] = s["ideal"]; agg["diag_heads"] = s.get("diag_heads", {})
    agg["timing_total_s"] = ms([s["timing"]["total_s"] for s in S])

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    bins = np.logspace(-0.5, 2.5, 70)
    for m, col in (("ista_banach", "C0"), ("ista_gap", "C1"), ("ista_rhoS", "C2")):
        r = np.concatenate([tr[f"ratios_{m}_cold"] for tr in TR if f"ratios_{m}_cold" in tr]); r = r[np.isfinite(r)]
        ax.hist(r, bins=bins, histtype="step", linewidth=3.2 if m == "ista_banach" else 1.4, color=col, label=f"{LABEL[m]} (median {np.median(r):.1f})")
    ax.axvline(1, color="k", lw=1); ax.axvline((1 + rho) / (1 - rho), color="r", ls="--", lw=1, label="$(1+\\rho)/(1-\\rho)$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("certificate / true error"); ax.set_ylabel("count (all seeds, all $k\\leq k^\\ast$)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig3_tightness.png", dpi=170, bbox_inches="tight"); fig.savefig(out / "fig3_tightness.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    order = ["cold", "cp_K8_xhat", "cp_K16_xhat"]
    data, labels, cols = [], [], []
    for st in order:
        for m, c in (("ista_banach", "C0"), ("fista_gap", "C3")):
            data.append(np.concatenate([tr[f"kstar_{m}_{st}"] for tr in TR]))
            labels.append(f"{SLAB[st]}\n{'ISTA' if m == 'ista_banach' else 'FISTA'}"); cols.append(c)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.6, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    ax.set_yscale("log"); ax.grid(alpha=0.25, which="both"); ax.tick_params(axis="x", labelsize=7)
    ax.set_ylabel("certified iterations $k^\\ast$ at $\\tau=10^{-3}$")
    fig.tight_layout(); fig.savefig(out / "fig4_iterations.png", dpi=170); fig.savefig(out / "fig4_iterations.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    it = np.arange(len(agg["E4"]["ista"]))
    for key, lab in (("ista", "ISTA (LASSO)"), ("fista", "FISTA (LASSO)"), ("enista", "ISTA (elastic net)"), ("enfista", "FISTA (elastic net)")):
        ax.plot(it[1:], agg["E4"][key][1:], lw=1.1, label=lab)
    for h, lab in (("cp_K16_xstar", "LISTA-CP to $x^\\star$, layers 1..16"), ("cp_K16_xhat", "LISTA-CP to $\\hat x_\\varepsilon$, layers 1..16")):
        ax.plot(np.arange(1, 17), agg["per_layer"][h], "s--", ms=3, lw=1, label=lab)
    Kof = {"cold": 0, "cp_K8_xhat": 8, "cp_K16_xhat": 16}
    for m, mk, c in (("ista_banach", "o", "C0"), ("fista_gap", "^", "C3")):
        for t, fill in zip(taus, ("none", c, "none")):
            xs = [Kof[st] + R[m][st][t]["k_star_mean"]["mean"] for st in order]; ys = [R[m][st][t]["nmse_out_db"]["mean"] for st in order]
            ax.plot(xs, ys, mk + "-", ms=5, lw=0.8, color=c, mfc=fill, label=f"certified {LABEL[m].split(' +')[0]}, $\\tau=10^{{{int(np.log10(float(t)))}}}$")
    ax.axhline(agg["nmse_xhat_xstar_db"]["mean"], color="gray", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("operator pairs (head layers + tail, certificate included)"); ax.set_ylabel("NMSE vs $x^\\star$ (dB)")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig5_pareto.png", dpi=170, bbox_inches="tight"); fig.savefig(out / "fig5_pareto.pdf", bbox_inches="tight"); plt.close(fig)

    keys = sorted([k for k in agg["E5"] if float(k) > 0], key=float); ex = np.array([float(k) for k in keys])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    axes[0].semilogx(ex, [agg["E5"][k]["rho"]["mean"] for k in keys], "o-", label="$\\rho$ (global)")
    axes[0].semilogx(ex, [agg["E5"][k]["rho_S_oracle_median"]["mean"] for k in keys], "s--", label="$\\rho_S$ at $\\mathrm{supp}(\\hat x_\\varepsilon)$ (median)")
    axes[0].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[0].set_ylabel("contraction constant"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)
    for key, lab, st in (("k", "ISTA + Banach, cold", "o-"), ("k_warm", "ISTA + Banach, K=8 head", "s--"), ("k_fista", "FISTA + gap, cold", "^-")):
        axes[1].loglog(ex, [agg["E5"][k][key]["0.001"]["mean"] for k in keys], st, label=lab)
    axes[1].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[1].set_ylabel("$k^\\ast$ at $\\tau=10^{-3}$"); axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1)
    axes[2].semilogx(ex, [agg["E5"][k]["nmse_xhat_db"]["mean"] for k in keys], "o-", label="elastic net")
    axes[2].axhline(agg["E5"]["0.0"]["nmse_xhat_db"]["mean"], color="gray", ls=":", label="LASSO ($\\varepsilon=0$)")
    axes[2].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[2].set_ylabel("NMSE$(\\hat x_\\varepsilon,x^\\star)$ (dB)"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig6_elastic_net.png", dpi=170, bbox_inches="tight"); fig.savefig(out / "fig6_elastic_net.pdf", bbox_inches="tight"); plt.close(fig)

    with open(root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=1)
    print(f"{len(S)} seeds; rho={rho:.5f}; E1 total {agg['E1_total']}")
    for m in METHODS:
        for st in R[m]:
            print(f"  {m:16s} {st:13s} " + "  ".join(f"{t}: {R[m][st][t]['k_star_mean']['mean']:6.1f}+-{R[m][st][t]['k_star_mean']['std']:4.1f}" for t in taus))
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
