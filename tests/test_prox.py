import numpy as np
import torch
from scipy.optimize import minimize_scalar

from cu.algos.prox import elastic_net_prox, elastic_net_prox_t, soft_threshold, soft_threshold_t

rng = np.random.default_rng(123)


def test_soft_threshold_closed_form():
    v = rng.standard_normal(10_000) * 3
    theta = 0.7
    x = soft_threshold(v, theta)
    on = x != 0
    assert np.allclose(v[on] - x[on], theta * np.sign(x[on]), atol=1e-14)
    assert np.all(np.abs(v[~on]) <= theta + 1e-14)
    assert soft_threshold(2.0, 0.5) == 1.5 and soft_threshold(-2.0, 0.5) == -1.5 and soft_threshold(0.3, 0.5) == 0.0


def test_elastic_net_prox_optimality_and_bruteforce():
    theta, c = 0.4, 0.3
    v = rng.standard_normal(2000) * 2
    x = elastic_net_prox(v, theta, c)
    on = x != 0
    assert np.allclose(v[on] - (1 + c) * x[on], theta * np.sign(x[on]), atol=1e-13)
    assert np.all(np.abs(v[~on]) <= theta + 1e-14)
    v5 = np.array([1.5, -0.2, 0.0, 3.0, -2.2])
    f1 = lambda z, vi: 0.5 * (z - vi) ** 2 + theta * abs(z) + 0.5 * c * z**2
    brute = np.array([minimize_scalar(f1, args=(vi,), bounds=(-10, 10), method="bounded", options=dict(xatol=1e-12)).x for vi in v5])
    assert np.allclose(brute, elastic_net_prox(v5, theta, c), atol=1e-6)


def test_numpy_equals_torch_float64():
    v = rng.standard_normal((64, 500)) * 2
    theta = rng.uniform(0.1, 1.0, size=(64, 1))
    c = 0.05
    a = elastic_net_prox(v, theta, c)
    b = elastic_net_prox_t(torch.from_numpy(v), torch.from_numpy(theta), c).numpy()
    assert np.array_equal(a, b) or np.allclose(a, b, atol=1e-16, rtol=0)
    a = soft_threshold(v, 0.3)
    b = soft_threshold_t(torch.from_numpy(v), 0.3).numpy()
    assert np.allclose(a, b, atol=1e-16, rtol=0)


def test_per_instance_threshold_broadcast():
    v = rng.standard_normal((8, 50))
    theta = np.linspace(0.1, 0.8, 8)[:, None]
    x = elastic_net_prox(v, theta, 0.0)
    for i in range(8):
        assert np.allclose(x[i], soft_threshold(v[i], theta[i, 0]))


def test_prox_is_nonexpansive():
    x = rng.standard_normal((5000, 100))
    z = x + rng.standard_normal((5000, 100)) * rng.uniform(0.01, 5, size=(5000, 1))
    for theta, c in [(0.3, 0.0), (0.3, 0.2), (2.0, 1.0)]:
        d_in = np.linalg.norm(x - z, axis=1)
        d_out = np.linalg.norm(elastic_net_prox(x, theta, c) - elastic_net_prox(z, theta, c), axis=1)
        assert np.all(d_out <= d_in * (1 + 1e-12) + 1e-15)
        assert np.all(d_out <= d_in / (1 + c) * (1 + 1e-12) + 1e-15)
