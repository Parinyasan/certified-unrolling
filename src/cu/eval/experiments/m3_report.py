from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cu.eval.metrics import worst_case_ratio


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/m3")
    ap.add_argument("--out", default="results/m3/figs")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(p for p in root.glob("seed*") if (p / "summary.json").exists())
    S = [json.load(open(p / "summary.json")) for p in seeds]
    TR = [np.load(p / "traces.npz") for p in seeds]
    print(f"{len(S)} seeds: {[s['seed'] for s in S]}")
    tols = [str(t) for t in S[0]["eps_tols"]]
    starts = list(S[0]["E3"].keys()); starts.remove("d_rel_median")
    rho = float(np.mean([s["rho"] for s in S]))
    agg = dict(n_seeds=len(S), seeds=[s["seed"] for s in S], rho=ms([s["rho"] for s in S]), eps_tols=tols, starts=starts,
               worst_case_ratio=worst_case_ratio(rho))

    agg["E1"] = dict(violations=int(sum(s["E1"]["violations"] for s in S)), checks=int(sum(s["E1"]["checks"] for s in S)))
    agg["heads"] = {h: {k: ms([s["heads"][h][k] for s in S]) for k in ("nmse_vs_xhat_db", "nmse_vs_xstar_db", "d_rel_median", "train_time_s")}
                    for h in S[0]["heads"]}
    agg["nmse_xhat_xstar_db"] = ms([s["nmse_xhat_xstar_db"] for s in S])
    agg["E3"] = {st: {t: {k: ms([s["E3"][st][t][k] for s in S]) for k in ("k_star_mean", "k_star_median", "k_star_max", "bound_mean", "nmse_out_db", "tail_time_s")}
                      for t in tols} for st in starts}
    for st in starts:
        for t in tols:
            agg["E3"][st][t]["bound_dominates_total"] = int(sum(s["E3"][st][t]["bound_dominates"] for s in S))
            agg["E3"][st][t]["speedup_vs_cold"] = ms([s["E3"]["cold"][t]["k_star_mean"] / s["E3"][st][t]["k_star_mean"] for s in S])
    agg["E2"] = dict(r_cold={q: ms([s["E2"]["r_cold"][q] for s in S]) for q in ("p50", "p90", "p99", "p100")},
                     frac_r_above_worst_case=ms([s["E2"]["frac_r_above_worst_case"] for s in S]),
                     rho_hat=dict(violations=int(sum(s["E2"]["rho_hat"]["violations"] for s in S)), checks=int(sum(s["E2"]["rho_hat"]["checks"] for s in S)),
                                  ratio_p50=ms([s["E2"]["rho_hat"]["ratio"]["p50"] for s in S])),
                     rho_S=dict(violations=int(sum(s["E2"]["rho_S"]["violations"] for s in S)), checks=int(sum(s["E2"]["rho_S"]["checks"] for s in S)),
                                ratio_p50=ms([s["E2"]["rho_S"]["ratio"]["p50"] for s in S]), ratio_p90=ms([s["E2"]["rho_S"]["ratio"]["p90"] for s in S]),
                                rho_S_median=ms([s["E2"]["rho_S"]["rho_S_median"] for s in S]),
                                k_support_stable_median=ms([s["E2"]["rho_S"]["k_support_stable_median"] for s in S]),
                                frac_support_stable_before_kstar=ms([s["E2"]["rho_S"]["frac_support_stable_before_kstar"] for s in S])))
    agg["E5"] = {}
    for e in S[0]["E5"]:
        ent = dict(rho=ms([s["E5"][e]["rho"] for s in S]), nmse_xhat_db=ms([s["E5"][e]["nmse_xhat_db"] for s in S]),
                   rho_S_median=ms([s["E5"][e]["rho_S_median"] for s in S]))
        if S[0]["E5"][e]["k_star"]:
            ent["k_star"] = {t: ms([s["E5"][e]["k_star"][t] for s in S]) for t in tols}
            ent["k_star_warm"] = {t: ms([s["E5"][e]["k_star_warm"][t] for s in S]) for t in tols}
        agg["E5"][e] = ent
    agg["float32"] = {t: dict(violations=int(sum(s["float32_ablation"][t]["violations"] for s in S)), checks=int(sum(s["float32_ablation"][t]["checks"] for s in S)),
                              frac_not_fired=ms([s["float32_ablation"][t]["frac_not_fired"] for s in S])) for t in tols}
    agg["timing_total_s"] = ms([s["timing"]["total_s"] for s in S]); agg["timing_sum_s"] = float(sum(s["timing"]["total_s"] for s in S))

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for lab, key, col in (("cold start", "cold", "C0"), ("warm start (LISTA-CP, K=8)", "cp_K8_xhat", "C1")):
        ce = np.concatenate([tr[f"{key}_certs"][np.arange(tr[f"{key}_certs"].shape[1])[None, :] < tr[f"{key}_k_star"][:, None]] for tr in TR])
        er = np.concatenate([tr[f"{key}_errs"][np.arange(tr[f"{key}_errs"].shape[1])[None, :] < tr[f"{key}_k_star"][:, None]] for tr in TR])
        ok = (ce > 0) & (er > 0)
        idx = np.flatnonzero(ok)
        if len(idx) > 200_000:
            idx = np.random.default_rng(0).choice(idx, 200_000, replace=False)
        ax.loglog(er[idx], ce[idx], ".", ms=1.5, alpha=0.25, color=col, label=f"{lab}: {ok.sum():,} points", rasterized=True)
    lo, hi = er[ok].min(), ce[ok].max(); g = np.array([lo, hi])
    ax.loglog(g, g, "k-", lw=1, label="$C_k=\\|x_k-\\hat x_\\varepsilon\\|$"); ax.loglog(g, worst_case_ratio(rho) * g, "r--", lw=1, label=f"$(1+\\rho)/(1-\\rho)={worst_case_ratio(rho):.0f}$")
    ax.set_xlabel("true error $\\|x_k-\\hat x_\\varepsilon\\|_2$"); ax.set_ylabel("certificate $C_k$"); ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"E1: {agg['E1']['violations']} violations in {agg['E1']['checks']:,} checks ({len(S)} seeds)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig2_e1_certificate_scatter.png", dpi=170); fig.savefig(out / "fig2_e1_certificate_scatter.pdf", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    r_cold = np.concatenate([tr["r_cold"] for tr in TR]); r_S = np.concatenate([tr["r_S"] for tr in TR]); r_hat = np.concatenate([tr["r_hat"] for tr in TR])
    bins = np.logspace(-1, np.log10(max(r_cold.max(), 300)), 80)
    ax.hist(r_cold, bins=bins, alpha=0.6, label=f"certified, $\\rho$ (median {np.median(r_cold):.1f})")
    ax.hist(r_S[np.isfinite(r_S)], bins=bins, alpha=0.6, label=f"support-local $\\rho_S$ (uncertified; median {np.median(r_S):.1f})")
    ax.hist(r_hat[np.isfinite(r_hat)], bins=bins, alpha=0.4, label=f"adaptive $\\hat\\rho_k$ (uncertified; median {np.nanmedian(r_hat):.2f})")
    ax.axvline(1.0, color="k", lw=1); ax.axvline(worst_case_ratio(rho), color="r", ls="--", lw=1, label="$(1+\\rho)/(1-\\rho)$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$r_k = C_k / \\|x_k-\\hat x_\\varepsilon\\|$"); ax.set_ylabel("count"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig3_e2_tightness.png", dpi=170); fig.savefig(out / "fig3_e2_tightness.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, len(tols), figsize=(3.4 * len(tols), 3.6), sharey=False)
    order = ["cold"] + [f"cp_K{K}_xhat" for K in (2, 4, 8, 16) if f"cp_K{K}_xhat" in starts]
    labels = ["cold"] + [f"K={K}" for K in (2, 4, 8, 16) if f"cp_K{K}_xhat" in starts]
    for ax, t in zip(np.atleast_1d(axes), tols):
        data = [np.concatenate([tr[f"kstar_{st}_{float(t)}"] for tr in TR]) for st in order]
        bnd = [np.mean([tr[f"bound_{st}_{float(t)}"].mean() for tr in TR]) for st in order]
        ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.6)
        ax.plot(np.arange(1, len(order) + 1), bnd, "r_", ms=14, mew=2, label="Theorem 2 bound (mean)")
        ax.set_yscale("log"); ax.set_title(f"$\\varepsilon_{{\\rm tol}} = 10^{{{int(np.log10(float(t)))}}}\\,\\|\\hat x_\\varepsilon\\|$", fontsize=9)
        ax.set_ylabel("certified tail iterations $k^\\ast$"); ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig4_e3_iterations.png", dpi=170); fig.savefig(out / "fig4_e3_iterations.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    n_it = len(S[0]["E4"]["ista"]) - 1
    it = np.arange(n_it + 1)
    for key, lab in (("ista", "ISTA"), ("fista", "FISTA"), ("enista", f"elastic-net ISTA ($\\varepsilon$)")):
        ax.plot(it[1:], np.mean([s["E4"][key] for s in S], 0)[1:], lw=1.2, label=lab)
    for hk, lab in (("cp_K16_xstar", "LISTA-CP trained to $x^\\star$ (layers 1..16)"), ("cp_K16_xhat", "LISTA-CP trained to $\\hat x_\\varepsilon$ (layers 1..16)")):
        if f"{hk}_per_layer" in S[0]["E4"]:
            ax.plot(np.arange(1, 17), np.mean([s["E4"][f"{hk}_per_layer"] for s in S], 0), "s--", ms=3, lw=1, label=lab)
    mk = dict(zip(tols, ["o", "^", "D"]))
    for t in tols:
        xs = [np.mean([s["E4"]["ours"][st][t]["ops"] for s in S]) for st in order]
        ys = [np.mean([s["E4"]["ours"][st][t]["nmse_db"] for s in S]) for st in order]
        ax.plot(xs, ys, mk[t] + "-", ms=5, lw=1, label=f"ours, $\\varepsilon_{{\\rm tol}}=10^{{{int(np.log10(float(t)))}}}$ (cold, K=2,4,8,16)")
    ax.axhline(agg["nmse_xhat_xstar_db"]["mean"], color="gray", ls=":", lw=1, label="NMSE$(\\hat x_\\varepsilon, x^\\star)$")
    ax.set_xscale("log"); ax.set_xlabel("operator applications (head layers + certified tail iterations)"); ax.set_ylabel("NMSE vs $x^\\star$ (dB)")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(out / "fig5_e4_pareto.png", dpi=170); fig.savefig(out / "fig5_e4_pareto.pdf"); plt.close(fig)

    eps_keys = sorted(agg["E5"], key=float); ex = np.array([float(e) for e in eps_keys]); pos = ex > 0
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].semilogx(ex[pos], [agg["E5"][e]["rho"]["mean"] for e in np.array(eps_keys)[pos]], "o-", label="$\\rho$ (global)")
    axes[0].semilogx(ex[pos], [agg["E5"][e]["rho_S_median"]["mean"] for e in np.array(eps_keys)[pos]], "s--", label="$\\rho_S$ (support-local, median)")
    axes[0].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[0].set_ylabel("contraction constant"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)
    for t in tols:
        axes[1].loglog(ex[pos], [agg["E5"][e]["k_star"][t]["mean"] for e in np.array(eps_keys)[pos]], "o-", label=f"cold, $10^{{{int(np.log10(float(t)))}}}$")
        axes[1].loglog(ex[pos], [agg["E5"][e]["k_star_warm"][t]["mean"] for e in np.array(eps_keys)[pos]], "s--", label=f"warm K=8, $10^{{{int(np.log10(float(t)))}}}$")
    axes[1].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[1].set_ylabel("$k^\\ast$ (mean)"); axes[1].legend(fontsize=6); axes[1].grid(alpha=0.25, which="both")
    nm = [agg["E5"][e]["nmse_xhat_db"]["mean"] for e in eps_keys]
    axes[2].semilogx(ex[pos], np.array(nm)[pos], "o-", label="elastic net $\\hat x_\\varepsilon$")
    axes[2].axhline(nm[0], color="gray", ls=":", label="LASSO ($\\varepsilon=0$)")
    axes[2].set_xlabel("$\\varepsilon/\\sigma_{\\max}^2$"); axes[2].set_ylabel("NMSE$(\\hat x_\\varepsilon, x^\\star)$ (dB)"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig6_e5_elastic_net.png", dpi=170); fig.savefig(out / "fig6_e5_elastic_net.pdf"); plt.close(fig)

    with open(root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=1)
    print(f"E1: {agg['E1']['violations']} / {agg['E1']['checks']:,} violations;  rho = {rho:.5f}")
    print("E3 k* mean+-std (speedup vs cold):")
    for st in starts:
        print(f"  {st:16s} " + "  ".join(f"tol {t}: {agg['E3'][st][t]['k_star_mean']['mean']:6.1f}+-{agg['E3'][st][t]['k_star_mean']['std']:4.1f} (x{agg['E3'][st][t]['speedup_vs_cold']['mean']:.2f})" for t in tols))
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
