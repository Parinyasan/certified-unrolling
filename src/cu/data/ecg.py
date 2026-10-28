from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pywt

from cu.ops.linear import DenseOperator, SubsampledOperator, compose

CACHE = Path(os.environ.get("CU_CACHE_DIR", "cache")) / "data" / "ecg"
FS = 360
DEFAULT_RECORDS = ("100", "101", "103", "112", "115", "117", "121", "123", "213", "220")
TEST_RECORDS = ("103", "117", "220")


PN_URLS = ("https://physionet-open.s3.amazonaws.com/{db}/1.0.0/{file}", "https://physionet.org/files/{db}/1.0.0/{file}")


def _fetch_record(name: str, pn_dir: str, channel: int = 0) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{pn_dir}_{name}_ch{channel}.npy"
    if f.exists():
        return np.load(f)
    import wfdb

    from cu.data.download import fetch

    raw = CACHE / "raw" / pn_dir
    fetch([u.format(db=pn_dir, file=f"{name}.hea") for u in PN_URLS], raw / f"{name}.hea")
    hdr = wfdb.rdheader(str(raw / name))
    for dat in sorted(set(hdr.file_name)):
        fetch([u.format(db=pn_dir, file=dat) for u in PN_URLS], raw / dat)
    rec = wfdb.rdrecord(str(raw / name))
    sig = np.asarray(rec.p_signal[:, channel], dtype=np.float64)
    np.save(f, sig)
    return sig


def wavelet_synthesis_matrix(n: int, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    w = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(np.zeros(n), w, mode="periodization", level=level)
    _, slices = pywt.coeffs_to_array(coeffs)
    Psi = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        c = pywt.array_to_coeffs(e, slices, output_format="wavedec")
        Psi[:, j] = pywt.waverec(c, w, mode="periodization")[:n]
    return Psi


def dct_synthesis_matrix(n: int) -> np.ndarray:
    from scipy.fft import idct

    return idct(np.eye(n), type=2, norm="ortho", axis=0)


def make_dictionary(n_sig: int, kind: str) -> np.ndarray:
    if kind == "db4":
        return wavelet_synthesis_matrix(n_sig, "db4", 5)
    if kind == "db4+dct":
        return np.hstack([wavelet_synthesis_matrix(n_sig, "db4", 5), dct_synthesis_matrix(n_sig)])
    raise ValueError(kind)


def _windows(sig: np.ndarray, n_sig: int, stride: int, rng: np.random.Generator, n_max: int) -> np.ndarray:
    starts = np.arange(0, len(sig) - n_sig, stride)
    if len(starts) > n_max:
        starts = np.sort(rng.choice(starts, n_max, replace=False))
    W = np.stack([sig[s: s + n_sig] for s in starts])
    return W - W.mean(axis=1, keepdims=True)


def load_ecg(cfg: dict, n_samples: int, seed: int, split: str = "train"):
    rng = np.random.default_rng(seed)
    n_sig = int(cfg.get("n_sig", 512))
    recs = tuple(cfg.get("records", DEFAULT_RECORDS))
    recs = [r for r in recs if (r in TEST_RECORDS) == (split == "test")]
    per_rec = int(np.ceil(n_samples / len(recs)))
    blocks = [_windows(_fetch_record(r, "mitdb"), n_sig, n_sig // 2, rng, per_rec) for r in recs]
    S = np.vstack(blocks)
    rec_ids = np.concatenate([[r] * len(b) for r, b in zip(recs, blocks)])
    perm = rng.permutation(len(S))[:n_samples]
    S = S[perm]; rec_ids = rec_ids[perm]

    kinds = cfg.get("noise", "bw+ma").split("+")
    noise_src = {k: _fetch_record(k, "nstdb") for k in kinds}
    N = np.zeros_like(S)
    for i in range(len(S)):
        wts = rng.dirichlet(np.ones(len(kinds)))
        for k, wt in zip(kinds, wts):
            src = noise_src[k]
            s0 = rng.integers(0, len(src) - n_sig)
            seg = src[s0: s0 + n_sig]
            seg = seg - seg.mean()
            N[i] += wt * seg / (np.linalg.norm(seg) + 1e-12)
    snr_db = float(cfg.get("snr_db", 10.0))
    N *= np.linalg.norm(S, axis=1, keepdims=True) / np.linalg.norm(N, axis=1, keepdims=True) * 10 ** (-snr_db / 20)

    Psi = make_dictionary(n_sig, cfg.get("dictionary", "db4"))
    meas = cfg.get("measurement", "identity")
    if meas == "identity":
        rows = np.arange(n_sig)
    elif meas == "subsample":
        m = int(round(float(cfg.get("m_frac", 0.5)) * n_sig))
        rows = np.sort(np.random.default_rng(int(cfg.get("mask_seed", 0))).choice(n_sig, m, replace=False))
    elif meas == "gaussian":
        rows = None
    else:
        raise ValueError(meas)
    if meas == "gaussian":
        m = int(round(float(cfg.get("m_frac", 0.5)) * n_sig))
        Mmat = np.random.default_rng(int(cfg.get("mask_seed", 0))).standard_normal((m, n_sig)) / np.sqrt(m)
        M = DenseOperator(Mmat)
        rows = None
    else:
        M = SubsampledOperator(rows, n_sig)
    A = compose(M, Psi)
    Y = M.matvec(S + N)
    X_star = np.linalg.lstsq(Psi, S.T, rcond=None)[0].T if Psi.shape[1] > Psi.shape[0] else S @ Psi
    lam = float(cfg.get("lam", 0.1)) * np.max(np.abs(A.rmatvec(Y)), axis=1)
    eps = float(cfg.get("eps_rel", 1e-2)) * A.sigma_max() ** 2
    meta = dict(source="ecg", fs=FS, n_sig=n_sig, records=recs, record_ids=rec_ids, Psi=Psi, rows=rows, S=S, N=N, lam=lam, eps=eps,
                snr_db=snr_db, dictionary=cfg.get("dictionary", "db4"), measurement=meas, split=split)
    return A, Y, X_star, meta
