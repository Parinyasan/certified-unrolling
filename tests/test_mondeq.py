import numpy as np
import torch

from cu.algos.fixed_point import compute_and_cache_xhat
from cu.models.mondeq_tail import MonDEQTail

torch.manual_seed(0)
rng = np.random.default_rng(0)


def _tail(scale):
    t = MonDEQTail(m_y=40, n_out=80, d=80, m=0.5, init_scale=scale, dtype=torch.float64)
    return t.numpy_tail()


def test_strong_monotonicity_and_rho():
    for scale in [0.05, 0.5, 2.0]:
        nt = _tail(scale)
        assert nt.strong_monotonicity_gap() >= -1e-10
        assert 0.0 <= nt.rho() < 1.0
        assert 0 < nt.alpha_ < 2 * nt.m / nt.L**2 + 1e-15
        Lip = np.linalg.norm(np.eye(nt.M.shape[0]) - nt.alpha_ * nt.M, 2)
        assert Lip <= nt.rho() + 1e-12


def test_certificate_holds_for_mondeq():
    B = 300
    for scale in [0.5, 2.0]:
        nt = _tail(scale)
        Y = rng.standard_normal((B, 40))
        zstar, n_it, fd = nt.solve_fixed_point(Y, tol=1e-12, max_iter=100000)
        assert np.all(fd <= 1e-12)
        z0 = rng.standard_normal((B, 80)) * (rng.random((B, 1)) < 0.5)
        tol = 1e-4 * np.maximum(np.linalg.norm(zstar, axis=1), 1e-8)
        res = nt.run_until(z0, Y, tol, xhat=zstar)
        slack = 10 * nt.certificate_factor() * 1e-12
        assert np.all(res.converged)
        assert not np.any(res.certs < res.errs_vs_xhat - slack), "MonDEQ certificate violated"
        assert np.all(np.linalg.norm(res.x - zstar, axis=1) <= tol + slack)
        xk = nt.readout(res.x, Y); xs = nt.readout(zstar, Y)
        assert np.all(np.linalg.norm(xk - xs, axis=1) <= nt.C_norm * tol + slack)


def test_jfb_forward_runs_and_matches_numpy_fixed_point():
    t = MonDEQTail(m_y=40, n_out=80, d=80, m=0.5, init_scale=0.5, dtype=torch.float64)
    Y = torch.as_tensor(rng.standard_normal((8, 40)))
    x, z, k = t(Y, max_iter=5000, tol=1e-10, return_k=True)
    nt = t.numpy_tail()
    zstar, _, _ = nt.solve_fixed_point(Y.numpy(), tol=1e-12)
    assert np.max(np.abs(z.detach().numpy() - zstar)) < 1e-8
    assert np.allclose(x.detach().numpy(), nt.readout(zstar, Y.numpy()), atol=1e-7)
    x.sum().backward()
    assert all(p.grad is not None for p in t.parameters())
