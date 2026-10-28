import numpy as np
import torch

from cu.algos.ista import ista
from cu.data.synthetic import make_synth
from cu.models.lista import ALISTA, LISTACP, LISTACPSS, TiedLISTACP, analytic_weight


def _setup(K=8, B=16):
    b = make_synth(B, 50, 100, 0.1, 30.0, 0.1, 1e-2, seed=1)
    mu = 1.0 / b.A.sigma_max() ** 2
    return b, mu


def test_lista_cp_init_equals_ista():
    b, mu = _setup()
    for cls in (LISTACP, TiedLISTACP):
        head = cls(b.A.to_dense(), 8, mu=mu, dtype=torch.float64)
        x_head = head(torch.as_tensor(b.Y), torch.as_tensor(b.lam)).detach().numpy()
        x_ista = ista(b.A, b.Y, b.lam, 8, mu=mu)
        assert np.max(np.abs(x_head - x_ista)) < 1e-10, cls.__name__
        xs = head(torch.as_tensor(b.Y), torch.as_tensor(b.lam), return_all=True)
        _, trace = ista(b.A, b.Y, b.lam, 8, mu=mu, return_trace=True)
        for k in range(8):
            assert np.max(np.abs(xs[k].detach().numpy() - trace[k + 1])) < 1e-10


def test_lista_cpss_reduces_to_cp_without_selection():
    b, mu = _setup()
    head = LISTACPSS(b.A.to_dense(), 4, mu=mu, dtype=torch.float64, p=0.0, p_max=0.0)
    x = head(torch.as_tensor(b.Y), torch.as_tensor(b.lam)).detach().numpy()
    assert np.max(np.abs(x - ista(b.A, b.Y, b.lam, 4, mu=mu))) < 1e-10
    head2 = LISTACPSS(b.A.to_dense(), 4, mu=mu, dtype=torch.float64)
    assert head2.n_select(0) == round(0.012 * 100) and head2.n_select(20) == 13


def test_alista_analytic_weight_and_init():
    b, mu = _setup()
    A = b.A.to_dense()
    W = analytic_weight(A, n_iter=300)
    assert np.allclose(np.diag(W.T @ A), 1.0, atol=1e-8)
    assert np.linalg.norm(W.T @ A - np.diag(np.diag(W.T @ A)), "fro") <= np.linalg.norm(A.T @ A / np.sum(A * A, 0) - np.eye(100), "fro") + 1e-9
    head = ALISTA(A, 3, mu=mu, dtype=torch.float64, W=W)
    x = head(torch.as_tensor(b.Y), torch.as_tensor(b.lam)).detach().numpy()
    x1 = np.sign(b.Y @ W) * np.maximum(np.abs(b.Y @ W) - mu * b.lam[:, None], 0)
    xs = head(torch.as_tensor(b.Y), torch.as_tensor(b.lam), return_all=True)
    assert np.max(np.abs(xs[0].detach().numpy() - x1)) < 1e-10
    assert x.shape == (16, 100)


def test_layer_params_and_set_A():
    b, mu = _setup()
    head = LISTACP(b.A.to_dense(), 4, mu=mu)
    assert len(head.layer_params(2)) == 2 and head.layer_params(2)[0] is head.W[2]
    A2 = b.A.to_dense() + 0.01
    head.set_A(A2)
    assert np.allclose(head.A.numpy(), A2.astype(np.float32))
