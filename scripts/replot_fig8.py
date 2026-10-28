import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/m5r/e7")
S = json.load(open(out / "summary.json"))
shifts = S["shifts"]; sh = np.array(shifts)


def agg(key):
    v = np.array([[e[key] for e in S["synthetic"][str(r)]] for r in shifts]); return v.mean(1), v.std(1)


fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
for key, lab, st in (("ours_nmse_db", "certified (head + tail), any tail", "o-"), ("lista16_adapt_nmse_db", "LISTA-CP K=16 alone, coupling A'", "s--"),
                     ("lista16_frozen_nmse_db", "LISTA-CP K=16 alone, coupling A (frozen)", "d:"), ("alista16_recomputed_nmse_db", "ALISTA K=16 alone, W(A') recomputed", "^--"),
                     ("ista16_nmse_db", "16 ISTA steps (no learning)", "x-."), ("nmse_xhat_db", "elastic-net solution of A'", "k-")):
    m, s = agg(key); axes[0].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.1, capsize=2, label=lab)
axes[0].set_xlabel(r"$\|\Delta A\|_2/\|A\|_2$ (%)"); axes[0].set_ylabel(r"NMSE vs $x^\star$ (dB)"); axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
for key, lab, st in (("ours_k", "ISTA + Banach, K=8 head", "o-"), ("cold_k", "ISTA + Banach, cold", "s--"), ("ours_fista_k", "FISTA + gap, K=8 head", "^-"), ("cold_fista_k", "FISTA + gap, cold", "v--")):
    m, s = agg(key); axes[1].errorbar(sh * 100, m, yerr=s, fmt=st, ms=4, lw=1.1, capsize=2, label=lab)
axes[1].set_xlabel(r"$\|\Delta A\|_2/\|A\|_2$ (%)"); axes[1].set_ylabel(r"certified iterations $k^\ast$ ($\tau=10^{-3}$)"); axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout(); fig.savefig(out / "fig8_operator_shift.png", dpi=170, bbox_inches="tight"); fig.savefig(out / "fig8_operator_shift.pdf", bbox_inches="tight")
print("wrote", out / "fig8_operator_shift.pdf")
