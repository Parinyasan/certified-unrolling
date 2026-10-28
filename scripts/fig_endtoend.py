import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/m6/synthetic")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/figs")
runs = [json.load(open(p)) for p in sorted(src.glob("seed*.json"))]
Ks = [k for k in ("K2", "K4", "K8", "K16") if all(k in r["starts"] for r in runs)]
Kv = [int(k[1:]) for k in Ks]
TAUS = [("0.01", r"$\tau=10^{-2}$", "#4C72B0"), ("0.001", r"$\tau=10^{-3}$", "#DD8452"), ("0.0001", r"$\tau=10^{-4}$", "#55A868")]
TAILS = [("ista_banach", "ISTA + Banach", "o-"), ("fista_gap", "FISTA + gap", "s--")]
PANELS = [("pairs", "operator pairs"), ("batch", "batched time"), ("latency", "latency (single input)")]


def net(r, k, tl, t, kind):
    if kind == "pairs":
        return r["starts"][k]["tails"][tl][t]["net_factor_pairs"]
    if kind == "batch":
        return r["starts"][k]["tails"][tl][t]["net_factor_time"]
    return r["starts"][k]["latency"]["tails"][tl][t]["net_factor_time"]


def ms(runs, k, tl, t, kind):
    if kind == "batch":
        return np.mean([r["starts"][k]["tails"][tl][t]["total_ms"] for r in runs])
    return np.mean([r["starts"][k]["latency"]["tails"][tl][t]["total_ms"] for r in runs])


fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), sharey=True)
for ax, (kind, title) in zip(axes, PANELS):
    if kind == "latency" and not all("latency" in r["starts"]["K8"] for r in runs):
        ax.set_title(title + " (n/a)", fontsize=9); continue
    for tl, tlab, st in TAILS:
        for t, tlab2, col in TAUS:
            v = np.array([[net(r, k, tl, t, kind) for k in Ks] for r in runs])
            centre = v.mean(0) if kind == "pairs" else np.array([ms(runs, "cold", tl, t, kind) / ms(runs, k, tl, t, kind) for k in Ks])
            ax.errorbar(Kv, centre, yerr=v.std(0), fmt=st, color=col, ms=4, lw=1.1, capsize=2,
                        mfc=("white" if tl == "fista_gap" else col), label=f"{tlab}, {tlab2}")
    ax.axhline(1.0, color="k", lw=0.9, ls=":")
    ax.set_xscale("log", base=2); ax.set_xticks(Kv); ax.set_xticklabels([str(k) for k in Kv])
    ax.set_xlabel("head depth $K$"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
axes[0].set_ylabel("net factor  cold / (head + tail)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.08, 1, 1))
dst.mkdir(parents=True, exist_ok=True)
fig.savefig(dst / "fig9_endtoend.pdf", bbox_inches="tight"); fig.savefig(dst / "fig9_endtoend.png", dpi=170, bbox_inches="tight")
print("wrote", dst / "fig9_endtoend.pdf", "from", len(runs), "seeds, depths", Kv)
