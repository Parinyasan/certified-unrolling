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
    ap.add_argument("--root", default="results/m5r")
    args = ap.parse_args()
    root = Path(args.root); (root / "figs").mkdir(parents=True, exist_ok=True)
    agg = dict(e6={})
    for ds in sorted((root / "e6").glob("*")):
        S = [json.load(open(p)) for p in sorted(ds.glob("seed*/summary.json"))]
        if not S:
            continue
        s0 = S[0]; taus = [str(t) for t in s0["taus"]]
        e = dict(n_seeds=len(S), m=s0["m"], n=s0["n"], rho=float(np.mean([s["rho"] for s in S])), metric=s0["metric"],
                 **({"n_train": s0["n_train"], "n_test": s0["n_test"]} if "n_train" in s0 else {}),
                 E1=dict(violations=int(sum(s["E1"]["violations"] for s in S)), checks=int(sum(s["E1"]["checks"] for s in S))),
                 train_min=ms([sum(h["train_time_s"] for h in s["heads"].values()) / 60 for s in S]), offline_min=ms([s["offline_xhat_train_s"] / 60 for s in S]),
                 heads={h: dict(d_rel_median=ms([s["heads"][h]["d_rel_median"] for s in S]), metric=ms([s["heads"][h]["metric"] for s in S]),
                                nmse_vs_xhat_db=ms([s["heads"][h]["nmse_vs_xhat_db"] for s in S])) for h in s0["heads"]},
                 results={})
        for meth in s0["results"]:
            e["results"][meth] = {}
            for st in s0["results"][meth]:
                e["results"][meth][st] = {t: {k: ms([s["results"][meth][st][t][k] for s in S]) for k in ("k_star_mean", "metric", "time_per_window_ms", "rel_err_max")} for t in taus}
        if s0["metric"] == "signal_nmse_db":
            e["nmse_input_db"] = ms([s["nmse_input_db"] for s in S]); e["nmse_xhat_db"] = ms([s["nmse_xhat_db"] for s in S])
        else:
            for k in ("fb_y", "fb_xhat", "fb_random", "kurt_y", "kurt_xhat", "f_res", "f_fault"):
                e[k] = ms([s[k] for s in S])
        agg["e6"][ds.name] = e
    e7p = root / "e7" / "summary.json"
    if e7p.exists():
        agg["e7"] = json.load(open(e7p))

    exs = {n: np.load(root / "e6" / n / "seed0" / "example.npz", allow_pickle=True) for n in ("ecg_compressive", "cwru_or007") if (root / "e6" / n / "seed0" / "example.npz").exists()}
    if exs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        if "ecg_compressive" in exs:
            e = exs["ecg_compressive"]; fs = float(e["fs"]); t = np.arange(len(e["s"])) / fs; ax = axes[0]
            ax.plot(t, e["noisy"], color="0.7", lw=0.8, label="clean + NSTDB noise (before sensing)")
            ax.plot(t, e["s"], "k", lw=1, label="clean $s$"); ax.plot(t, e["recon"], "C3", lw=1, label=f"certified reconstruction ($k^\\ast={int(e['k_star'])}$)")
            ax.set_xlabel("time (s)"); ax.set_ylabel("mV"); ax.set_title(f"ECG record {e['record']}, Gaussian CS $m/n=1/2$, 20 dB", fontsize=9); ax.legend(fontsize=7)
        if "cwru_or007" in exs:
            e = exs["cwru_or007"]; fs = float(e["fs"]); t = 1e3 * np.arange(len(e["y"])) / fs; ax = axes[1]
            ax.plot(t, e["y"], color="0.6", lw=0.7, label="vibration $y$")
            xo = e["x_out"]; ax.plot(t, xo / max(1e-12, np.abs(xo).max()) * np.abs(e["y"]).max(), "C3", lw=0.9, label=f"recovered impulse train (scaled), $k^\\ast={int(e['k_star'])}$")
            T = 1e3 / float(e["f_fault"]); off = t[np.argmax(np.abs(xo))] % T
            for j in range(int(t[-1] / T) + 1):
                ax.axvline(j * T + off, color="C2", lw=0.6, alpha=0.5)
            ax.set_xlabel("time (ms)"); ax.set_title(f"CWRU 130 (outer race), BPFO = {float(e['f_fault']):.1f} Hz", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(root / "figs" / "fig7_real_examples.png", dpi=170); fig.savefig(root / "figs" / "fig7_real_examples.pdf"); plt.close(fig)
    with open(root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=1, default=float)
    for n, e in agg["e6"].items():
        print(n, f"seeds {e['n_seeds']} rho {e['rho']:.4f} E1 {e['E1']} k*(1e-3) " + " ".join(f"{m}/{st}={e['results'][m][st]['0.001']['k_star_mean']['mean']:.1f}" for m in e["results"] for st in e["results"][m]))


if __name__ == "__main__":
    main()
