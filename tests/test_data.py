import os
from pathlib import Path

import numpy as np
import pytest

from cu.data.synthetic import load_synthetic

CACHE = Path(os.environ.get("CU_CACHE_DIR", "cache")) / "data"


def test_synthetic_loader_shapes_and_snr():
    cfg = dict(m=250, n=500, A={"type": "gaussian"}, sparsity_p=0.1, snr_db=30, lam=0.1, eps_rel=1e-2)
    A, Y, X, meta = load_synthetic(cfg, 400, seed=3)
    assert A.shape == (250, 500) and Y.shape == (400, 250) and X.shape == (400, 500)
    assert np.allclose(np.linalg.norm(A.to_dense(), axis=0), 1.0)
    snr = 10 * np.log10(np.sum(A.matvec(X) ** 2, 1) / np.sum((Y - A.matvec(X)) ** 2, 1))
    assert abs(snr.mean() - 30) < 0.5
    assert meta["lam"].shape == (400,) and meta["eps"] > 0
    _, Y2, _, _ = load_synthetic(cfg, 400, seed=3)
    assert np.array_equal(Y, Y2)


@pytest.mark.skipif(not (CACHE / "ecg").exists(), reason="ECG cache absent (needs network)")
def test_ecg_loader():
    from cu.data.ecg import load_ecg

    A, Y, X, meta = load_ecg(dict(dictionary="db4", measurement="subsample", m_frac=0.5, snr_db=10), 50, 0, "test")
    assert A.shape == (256, 512) and Y.shape == (50, 256) and X.shape == (50, 512)
    snr = 10 * np.log10(np.sum(meta["S"] ** 2, 1) / np.sum(meta["N"] ** 2, 1))
    assert np.allclose(snr, 10.0, atol=1e-6)
    assert np.allclose(meta["S"], X @ meta["Psi"].T, atol=1e-10)
    assert set(meta["records"]) <= {"103", "117", "220"}


@pytest.mark.skipif(not (CACHE / "cwru").exists(), reason="CWRU cache absent (needs network)")
def test_cwru_loader():
    from cu.data.cwru import fault_band_energy_ratio, load_cwru

    A, Y, X, meta = load_cwru(dict(file="130"), 20, 0, "test")
    assert X is None and A.shape == (1024, 1024) and Y.shape == (20, 1024)
    assert meta["source"] == "cwru" and abs(meta["f_fault"] - 107.36) < 0.1
    r = fault_band_energy_ratio(Y, meta["f_fault"])
    assert r.shape == (20,) and np.all((r >= 0) & (r <= 1))
