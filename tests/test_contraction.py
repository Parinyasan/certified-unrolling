import numpy as np
import pytest

from cu.algos.tail import ContractiveTail
from cu.ops.linear import IdentityOperator
from tests.conftest import N_INSTANCES, SEED

rng = np.random.default_rng(SEED + 1)


def _ratios(tail, Y, lam, X, Xp):
    TX = tail.step(X, Y, lam)
    TXp = tail.step(Xp, Y, lam)
    num = np.linalg.norm(TX - TXp, axis=1)
    den = np.linalg.norm(X - Xp, axis=1)
    return num / den


def test_lemma1_closed_form_matches_spectral_norm(problem):
    A, lam, eps = problem["A"], problem["lam"], problem["eps"]
    tail = ContractiveTail(A, lam, eps)
    assert abs(tail.lipschitz_G() - A.gram_spectral_norm(tail.mu)) <= 1e-12
    assert abs(tail.rho() - A.gram_spectral_norm(tail.mu) / (1 + tail.mu * eps)) <= 1e-12
    for mu in [0.3 / A.sigma_max() ** 2, 1.0 / A.sigma_max() ** 2, 1.7 / A.sigma_max() ** 2]:
        t = ContractiveTail(A, lam, eps, mu=mu)
        assert abs(t.lipschitz_G() - A.gram_spectral_norm(mu)) <= 1e-12
    if A.sigma_min() == 0.0:
        assert np.isclose(tail.mu, 2.0 / A.sigma_max() ** 2)
        assert np.isclose(tail.lipschitz_G(), 1.0)
        assert np.isclose(tail.rho(), 1.0 / (1.0 + tail.mu * eps))
    else:
        assert np.isclose(tail.mu, 2.0 / (A.sigma_min() ** 2 + A.sigma_max() ** 2))
        kappa = A.sigma_max() / A.sigma_min()
        assert np.isclose(tail.lipschitz_G(), (kappa**2 - 1) / (kappa**2 + 1))
    assert tail.rho() < 1.0


def test_empirical_lipschitz_random_pairs(problem):
    A, Y, lam, eps = problem["A"], problem["Y"], problem["lam"], problem["eps"]
    tail = ContractiveTail(A, lam, eps)
    rho = tail.rho()
    B = N_INSTANCES
    n = A.n
    worst = 0.0
    n_pairs = 0
    for scale in [1e-3, 1e-1, 1.0, 10.0]:
        X = rng.standard_normal((B, n))
        Xp = X + scale * rng.standard_normal((B, n))
        r = _ratios(tail, Y, lam, X, Xp)
        worst = max(worst, r.max())
        n_pairs += B
    X = problem["X_star"] + 0.05 * rng.standard_normal((B, n))
    Xp = X * (rng.random((B, n)) < 0.5) + 0.02 * rng.standard_normal((B, n))
    r = _ratios(tail, Y, lam, X, Xp)
    worst = max(worst, r.max())
    n_pairs += B
    assert n_pairs == 5 * B
    assert worst <= rho * (1 + 1e-10), f"empirical Lipschitz {worst} exceeds rho {rho}"


def test_sharpness_affine_regime(problem):
    A, Y, lam, eps = problem["A"], problem["Y"], problem["lam"], problem["eps"]
    tail = ContractiveTail(A, lam, eps)
    Ad = A.to_dense()
    M = np.eye(A.n) - tail.mu * Ad.T @ Ad
    w, V = np.linalg.eigh(M)
    v = V[:, np.argmax(np.abs(w))]
    assert np.isclose(abs(w[np.argmax(np.abs(w))]), tail.lipschitz_G(), atol=1e-12)
    B = 64
    X = 1e4 * np.sign(rng.standard_normal((B, A.n))) + rng.standard_normal((B, A.n))
    delta = 1.0
    Xp = X + delta * v[None, :]
    G = lambda Z: Z - tail.mu * A.rmatvec(A.matvec(Z) - Y[:B])
    theta = tail.mu * lam[:B, None]
    affine = np.all(np.abs(G(X)) > theta + delta, axis=1) & np.all(np.abs(G(Xp)) > theta + delta, axis=1)
    assert affine.sum() >= B // 2, "too few pairs in the affine regime"
    r = _ratios(tail, Y[:B], lam[:B], X, Xp)
    assert np.allclose(r[affine], tail.rho(), rtol=1e-8), f"ratios {r[affine].min()}..{r[affine].max()} vs rho {tail.rho()}"
    assert np.all(r <= tail.rho() * (1 + 1e-10))


def test_edge_cases():
    rng2 = np.random.default_rng(7)
    A = rng2.standard_normal((20, 40))
    t = ContractiveTail(A, 0.1, eps=0.0)
    assert np.isclose(t.rho(), 1.0) and not t.is_contraction()
    with pytest.raises(ValueError):
        t.run_until(None, rng2.standard_normal(20), 1e-3)
    t2 = ContractiveTail(A, 0.1, eps=0.5)
    assert t2.rho() < 1.0 and np.isclose(t2.rho(), 1.0 / (1 + t2.mu * 0.5))
    I = IdentityOperator(30)
    t3 = ContractiveTail(I, 0.1, eps=0.0)
    assert np.isclose(t3.mu, 1.0) and np.isclose(t3.rho(), 0.0)
    y = rng2.standard_normal(30)
    res = t3.run_until(None, y, 1e-9)
    assert res.k_star == 1 and np.allclose(res.x, np.sign(y) * np.maximum(np.abs(y) - 0.1, 0))
