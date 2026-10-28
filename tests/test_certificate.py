import os

import numpy as np

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.algos.tail import ContractiveTail
from cu.eval.metrics import theorem2_bound
from tests.conftest import N_INSTANCES, SEED

XHAT_TOL = 1e-12
EPS_TOL_REL = 1e-4
CACHE_DIR = os.environ.get("CU_CACHE_DIR", "cache") + "/xhat"


def _x0_mixture(xhat, rng):
    B, n = xhat.shape
    x0 = np.zeros_like(xhat)
    kind = rng.integers(0, 3, size=B)
    scale = np.linalg.norm(xhat, axis=1, keepdims=True)
    x0[kind == 1] = rng.standard_normal((np.sum(kind == 1), n)) * scale[kind == 1] / np.sqrt(n)
    x0[kind == 2] = xhat[kind == 2] + 0.1 * scale[kind == 2] / np.sqrt(n) * rng.standard_normal((np.sum(kind == 2), n))
    return x0, kind


def test_certificate_never_violated(problem):
    A, Y, lam, eps = problem["A"], problem["Y"], problem["lam"], problem["eps"]
    tail = ContractiveTail(A, lam, eps)
    rho = tail.rho()
    xhat, info = compute_and_cache_xhat(Y, A, lam, eps, mu=tail.mu, tol=XHAT_TOL, cache_dir=CACHE_DIR, verbose=True)
    assert np.all(info["final_diff"] <= XHAT_TOL)
    assert np.all(np.linalg.norm(tail.step(xhat, Y) - xhat, axis=1) <= XHAT_TOL * 1.01)

    rng = np.random.default_rng(SEED + 2)
    x0, kind = _x0_mixture(xhat, rng)
    eps_tol = EPS_TOL_REL * np.linalg.norm(xhat, axis=1)
    res = tail.run_until(x0, Y, eps_tol, xhat=xhat, max_iter=100_000)
    assert np.all(res.converged)

    slack = 10.0 * tail.certificate_factor() * XHAT_TOL
    certs, errs = res.certs, res.errs_vs_xhat
    viol = certs < errs - slack
    n_checks = certs.size
    print(f"[{problem['name']}] rho={rho:.6f} B={Y.shape[0]} k_max={certs.shape[1]} checks={n_checks} "
          f"violations={int(viol.sum())} k*={res.k_star.mean():.1f} (min {res.k_star.min()}, max {res.k_star.max()}) "
          f"tail time {res.time_s:.1f}s")
    assert n_checks >= 10 * N_INSTANCES
    assert not viol.any(), f"{int(viol.sum())} certificate violations; worst gap {(errs - certs).max():.3e}"

    final_err = np.linalg.norm(res.x - xhat, axis=1)
    assert np.all(final_err <= eps_tol + slack)
    first = np.argmax(certs <= eps_tol[:, None], axis=1) + 1
    assert np.array_equal(first, res.k_star)
    assert set(np.unique(kind)) == {0, 1, 2}

    d = np.linalg.norm(x0 - xhat, axis=1)
    bound = theorem2_bound(rho, d, eps_tol)
    assert np.all(bound >= res.k_star), f"Theorem 2 bound violated in {np.sum(bound < res.k_star)} cases"


def test_certificate_single_instance_api(problem):
    A, Y, lam, eps = problem["A"], problem["Y"], problem["lam"], problem["eps"]
    tail = ContractiveTail(A, lam[0], eps)
    xhat, _ = compute_and_cache_xhat(Y[0], A, lam[0], eps, mu=tail.mu, tol=XHAT_TOL, cache_dir=CACHE_DIR)
    res = tail.run_until(None, Y[0], 1e-3 * np.linalg.norm(xhat), xhat=xhat)
    slack = 10.0 * tail.certificate_factor() * XHAT_TOL
    assert res.certs.ndim == 1 and res.certs.shape[0] == res.k_star
    assert np.all(res.certs >= res.errs_vs_xhat - slack)
    assert np.linalg.norm(res.x - xhat) <= 1e-3 * np.linalg.norm(xhat) + slack
    assert np.isclose(tail.certificate(res.x, res.x), 0.0)
