from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cu.ops.linear import ConvOperator

CACHE = Path(os.environ.get("CU_CACHE_DIR", "cache")) / "data" / "cwru"
FS = 12_000
BASE_URL = "https://engineering.case.edu/sites/default/files/{id}.mat"
FILES = {"97": ("normal", 1797), "105": ("IR007", 1797), "118": ("B007", 1797), "130": ("OR007@6", 1797)}
MULT = dict(BPFI=5.4152, BPFO=3.5848, FTF=0.39828, BSF=2.3568)
FAULT_OF = {"IR007": "BPFI", "OR007@6": "BPFO", "B007": "BSF", "normal": None}


def _download(fid: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    from cu.data.download import fetch

    return fetch(BASE_URL.format(id=fid), CACHE / f"{fid}.mat")


def load_raw(fid: str) -> np.ndarray:
    from scipy.io import loadmat

    d = loadmat(_download(fid))
    key = [k for k in d if k.endswith("_DE_time")][0]
    return np.asarray(d[key], dtype=np.float64).ravel()


def damped_sinusoid_atom(f_res: float, tau: float, fs: int = FS, n_atom: int | None = None) -> np.ndarray:
    n_atom = int(round(5 * tau * fs)) if n_atom is None else n_atom
    t = np.arange(1, n_atom + 1) / fs
    h = np.exp(-t / tau) * np.sin(2 * np.pi * f_res * t)
    return h / np.linalg.norm(h)


def estimate_resonance(sig: np.ndarray, fs: int = FS, band=(500.0, 5500.0)) -> float:
    from scipy.signal import welch

    f, P = welch(sig, fs=fs, nperseg=4096)
    ok = (f >= band[0]) & (f <= band[1])
    return float(f[ok][np.argmax(P[ok])])


def synthetic_bearing(n_total: int, rpm: float, fault: str, f_res: float, tau: float, snr_db: float, rng) -> np.ndarray:
    fr = rpm / 60.0
    f_fault = MULT[fault] * fr
    period = FS / f_fault
    n_imp = int(np.ceil(n_total / period))
    t_imp = np.arange(n_imp) * period + rng.normal(0, 0.01 * period, size=n_imp)
    x = np.zeros(n_total)
    for ti in t_imp:
        i = int(round(ti))
        if 0 <= i < n_total:
            x[i] = 1.0 + 0.2 * rng.standard_normal()
    h = damped_sinusoid_atom(f_res, tau)
    s = np.convolve(x, h)[:n_total]
    noise = rng.standard_normal(n_total)
    noise *= np.linalg.norm(s) / np.linalg.norm(noise) * 10 ** (-snr_db / 20)
    return s + noise


def load_cwru(cfg: dict, n_samples: int, seed: int, split: str = "train", allow_synthetic: bool = False):
    rng = np.random.default_rng(seed)
    fid = str(cfg.get("file", "130"))
    label, rpm = FILES[fid]
    n_sig = int(cfg.get("n_sig", 1024))
    source = "cwru"
    try:
        sig = load_raw(fid)
    except Exception as e:
        if not allow_synthetic:
            raise
        source = f"synthetic_bearing(fallback: {type(e).__name__})"
        f_res0 = float(cfg.get("f_res") or 3000.0)
        sig = synthetic_bearing(int(cfg.get("n_total_synth", 240_000)), rpm, FAULT_OF[label] or "BPFO", f_res0,
                                float(cfg.get("tau_ms", 0.5)) * 1e-3, float(cfg.get("snr_db", 10.0)), rng)
    sig = sig - sig.mean()
    sig = sig / sig.std()
    cut = int(0.7 * len(sig))
    seg = sig[:cut] if split == "train" else sig[cut:]
    hop = int(cfg.get("hop", n_sig // 2))
    starts = np.arange(0, len(seg) - n_sig, hop)
    if len(starts) > n_samples:
        starts = np.sort(rng.choice(starts, n_samples, replace=False))
    Y = np.stack([seg[s: s + n_sig] for s in starts])
    Y = Y - Y.mean(axis=1, keepdims=True)

    f_res = float(cfg["f_res"]) if cfg.get("f_res") else estimate_resonance(sig)
    tau = float(cfg.get("tau_ms", 0.5)) * 1e-3
    h = damped_sinusoid_atom(f_res, tau)
    A = ConvOperator(h, n_sig, center=0)
    lam = float(cfg.get("lam", 0.1)) * np.max(np.abs(A.rmatvec(Y)), axis=1)
    eps = float(cfg.get("eps_rel", 1e-2)) * A.sigma_max() ** 2
    fr = rpm / 60.0
    fault = FAULT_OF[label]
    meta = dict(source=source, fs=FS, file=fid, label=label, rpm=rpm, f_shaft=fr, fault=fault,
                f_fault=(MULT[fault] * fr if fault else None), f_res=f_res, tau=tau, atom=h,
                n_sig=n_sig, lam=lam, eps=eps, split=split)
    return A, Y, None, meta


def fault_band_energy_ratio(x: np.ndarray, f_fault: float, fs: int = FS, n_harm: int = 5, tol_hz: float = 3.0) -> np.ndarray:
    x = np.atleast_2d(np.abs(x))
    n = x.shape[1]
    F = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (f > 0) & (f <= (n_harm + 0.5) * f_fault)
    mask = np.zeros_like(f, dtype=bool)
    for k in range(1, n_harm + 1):
        mask |= np.abs(f - k * f_fault) <= tol_hz + 0.5 * fs / n
    return F[:, mask & band].sum(axis=1) / (F[:, band].sum(axis=1) + 1e-300)


def kurtosis(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    xc = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(xc**2, axis=1); m4 = np.mean(xc**4, axis=1)
    return m4 / (m2**2 + 1e-300) - 3.0
