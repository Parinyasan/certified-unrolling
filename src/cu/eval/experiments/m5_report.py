from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/m5")
    args = ap.parse_args()
    root = Path(args.root); (root / "figs").mkdir(parents=True, exist_ok=True)
    e6 = {p.parent.parent.name: json.load(open(p)) for p in sorted((root / "e6").glob("*/seed*/summary.json"))}
    agg = dict(e6=e6)
    e7p = root / "e7" / "summary.json"
    if e7p.exists():
        agg["e7"] = json.load(open(e7p))

    exs = {}
    for name in ("ecg_compressive", "cwru_or007"):
        p = root / "e6" / name / "seed0" / "example.npz"
        if p.exists():
            exs[name] = np.load(p, allow_pickle=True)
    if exs:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4))
        if "ecg_compressive" in exs:
            e = exs["ecg_compressive"]; fs = float(e["fs"]); t = np.arange(len(e["s"])) / fs
            ax = axes[0, 0]
            ax.plot(t, e["noisy"], color="0.7", lw=0.8, label="clean + NSTDB noise (before sensing)")
            ax.plot(t, e["s"], "k", lw=1, label="clean $s$")
            ax.plot(t, e["recon"], "C3", lw=1, label="$\\Psi x_{\\rm out}$ (ours, certified)")
            ax.set_xlabel("time (s)"); ax.set_ylabel("mV"); ax.set_title(f"ECG, Gaussian CS m/n = 1/2, 20 dB (MIT-BIH test record)", fontsize=9); ax.legend(fontsize=7)
            ax = axes[0, 1]; k = np.arange(1, len(e["certs"]) + 1)
            ax.semilogy(k, e["certs"], "C0", label="certificate $C_k$"); ax.semilogy(k, e["errs"], "C1--", label="true error $\\|x_k-\\hat x_\\varepsilon\\|$")
            ax.axvline(int(e["k_star"]), color="k", ls=":", lw=1, label=f"stop $k^\\ast={int(e['k_star'])}$")
            ax.set_xlabel("tail iteration $k$ (after K = 8 head layers)"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
        if "cwru_or007" in exs:
            e = exs["cwru_or007"]; fs = float(e["fs"]); t = 1e3 * np.arange(len(e["y"])) / fs
            ax = axes[1, 0]
            ax.plot(t, e["y"], color="0.6", lw=0.7, label="vibration $y$ (CWRU 130, OR fault)")
            ax.plot(t, e["x_out"] / max(1e-12, np.abs(e["x_out"]).max()) * np.abs(e["y"]).max(), "C3", lw=0.9, label="recovered impulse train $x_{\\rm out}$ (scaled)")
            f_fault = float(e["f_fault"]); T = 1e3 / f_fault
            for j in range(int(t[-1] / T) + 1):
                ax.axvline(j * T + (t[np.argmax(np.abs(e["x_out"]))] % T), color="C2", lw=0.6, alpha=0.5)
            ax.set_xlabel("time (ms)"); ax.set_title(f"CWRU outer-race fault; BPFO = {f_fault:.1f} Hz (green grid: fault period)", fontsize=9); ax.legend(fontsize=7)
            ax = axes[1, 1]; k = np.arange(1, len(e["certs"]) + 1)
            ax.semilogy(k, e["certs"], "C0", label="certificate $C_k$"); ax.semilogy(k, e["errs"], "C1--", label="true error")
            ax.axvline(int(e["k_star"]), color="k", ls=":", lw=1, label=f"stop $k^\\ast={int(e['k_star'])}$")
            ax.set_xlabel("tail iteration $k$"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(root / "figs" / "fig7_real_examples.png", dpi=170); fig.savefig(root / "figs" / "fig7_real_examples.pdf"); plt.close(fig)

    print(f"{'dataset':26s} {'A':>10s} {'rho':>8s} {'viol/checks':>14s} {'k* cold':>8s} {'k* warm':>8s} {'ms/inst':>8s} {'metric (xhat | head | ours)':>34s}")
    for name, s in e6.items():
        c, w = s["E3"]["cold"]["0.001"], s["E3"]["warm"]["0.001"]
        if s["kind"] == "ecg":
            met = f"NMSE dB {s['nmse_xhat_signal_db']:.2f} | {s['head']['nmse_signal_db']:.2f} | {w['nmse_signal_db']:.2f} (input {s['nmse_input_signal_db']:.1f})"
        else:
            met = f"fault-band {s['fault_band_ratio_xhat']:.3f} | {s['head']['fault_band_ratio']:.3f} | {w['fault_band_ratio']:.3f} (y {s['fault_band_ratio_y']:.3f}, null {s.get('fault_band_ratio_random_impulses', float('nan')):.3f})"
        print(f"{name:26s} {s['m']:>4d}x{s['n']:<5d} {s['rho']:8.4f} {s['E1']['violations']:>5d}/{s['E1']['checks']:<8d} {c['k_star_mean']:8.1f} {w['k_star_mean']:8.1f} {w['time_per_instance_ms']:8.2f} {met}")
    with open(root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=1, default=float)


if __name__ == "__main__":
    main()
