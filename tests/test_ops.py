import numpy as np

from cu.ops.linear import ConvOperator, DenseOperator, SubsampledOperator, compose

rng = np.random.default_rng(5)


def _check_op(A, n_trials=3):
    Ad = A.to_dense()
    m, n = A.shape
    assert Ad.shape == (m, n)
    X = rng.standard_normal((n_trials, n))
    Y = rng.standard_normal((n_trials, m))
    assert np.allclose(A.matvec(X), X @ Ad.T, atol=1e-12)
    assert np.allclose(A.rmatvec(Y), Y @ Ad, atol=1e-12)
    assert np.allclose(A.matvec(X[0]), Ad @ X[0], atol=1e-12)
    assert np.allclose(np.sum(A.matvec(X) * Y, axis=1), np.sum(X * A.rmatvec(Y), axis=1), atol=1e-10)
    s = np.linalg.svd(Ad, compute_uv=False)
    assert np.isclose(A.sigma_max(), s[0])
    assert np.isclose(A.sigma_min(), 0.0 if m < n else s[-1])


def test_subsampled_identity():
    rows = np.sort(rng.choice(64, 30, replace=False))
    _check_op(SubsampledOperator(rows, 64))


def test_subsampled_of_dense():
    B = DenseOperator(rng.standard_normal((40, 64)))
    rows = np.sort(rng.choice(40, 20, replace=False))
    _check_op(SubsampledOperator(rows, 64, base=B))


def test_conv_matches_numpy_convolve():
    h = rng.standard_normal(9)
    A = ConvOperator(h, 50, center=0)
    x = rng.standard_normal(50)
    assert np.allclose(A.matvec(x), np.convolve(x, h)[:50])
    _check_op(A)
    A2 = ConvOperator(h, 50)
    assert np.allclose(A2.matvec(x), np.convolve(x, h)[4:54])


def test_compose():
    M = SubsampledOperator(np.arange(0, 64, 2), 64)
    Psi = np.linalg.qr(rng.standard_normal((64, 64)))[0]
    A = compose(M, Psi)
    assert A.shape == (32, 64)
    assert np.allclose(A.to_dense(), Psi[::2])
    _check_op(A)
