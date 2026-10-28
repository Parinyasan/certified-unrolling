from __future__ import annotations

import os

import numpy as np
import pytest

from cu.data.synthetic import add_noise, conditioned_operator, gaussian_operator, lam_from_y, sparse_signals

N_INSTANCES = int(os.environ.get("CU_TEST_N", "10000"))
SEED = int(os.environ.get("CU_TEST_SEED", "0"))

M0_CONFIGS = {
    "underdet_250x500_eps1e-2": dict(kind="gaussian", m=250, n=500, p=0.10, snr_db=30.0, lam_rel=0.1, eps_rel=1e-2),
    "square_500x500_kappa3_eps0": dict(kind="conditioned", m=500, n=500, kappa=3.0, p=0.10, snr_db=30.0, lam_rel=0.1, eps_rel=0.0),
}


def build_problem(name: str, B: int, seed: int = SEED):
    cfg = M0_CONFIGS[name]
    rng = np.random.default_rng(seed)
    if cfg["kind"] == "gaussian":
        A = gaussian_operator(cfg["m"], cfg["n"], rng)
    else:
        A = conditioned_operator(cfg["n"], cfg["kappa"], rng, m=cfg["m"])
    X = sparse_signals(B, A.n, cfg["p"], rng)
    Y = add_noise(A.matvec(X), cfg["snr_db"], rng)
    lam = lam_from_y(A, Y, cfg["lam_rel"])
    eps = cfg["eps_rel"] * A.sigma_max() ** 2
    return dict(A=A, Y=Y, X_star=X, lam=lam, eps=eps, name=name, cfg=cfg, seed=seed)


@pytest.fixture(scope="session", params=list(M0_CONFIGS))
def problem(request):
    return build_problem(request.param, N_INSTANCES)
