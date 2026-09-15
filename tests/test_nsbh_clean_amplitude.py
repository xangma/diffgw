import torch

from torchwave.approximants.nsbh_clean_amplitude import delta2_prime_plus, gamma1_prime_small_lambda, outer_pn_center


def test_delta2_plus_limits_and_batch_shape():
    fr = torch.ones(4, dtype=torch.float64)
    ft = torch.tensor([0.0, 0.5, 1.0, 100.0], dtype=torch.float64)
    out = delta2_prime_plus(ft, fr)
    assert out.shape == ft.shape
    assert torch.all(torch.isfinite(out))
    assert out[0] < 1e-10
    assert abs(float(out[-1]) - 0.81248) < 1e-10
    assert torch.all(out[1:] > out[:-1])


def test_delta2_plus_vector_gradient():
    ft = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64, requires_grad=True)
    fr = torch.ones(3, dtype=torch.float64)
    y = delta2_prime_plus(ft, fr).sum()
    (grad,) = torch.autograd.grad(y, ft)
    assert torch.all(torch.isfinite(grad))
    assert torch.all(grad > 0)


def test_delta2_plus_matches_public_callable_at_fixed_ratios():
    ls = __import__("pytest").importorskip("lalsimulation")
    for ratio in (0.5, 1.0, 2.0, 10.0):
        ft, fr = ratio * 1000.0, 1000.0
        expected = ls.SimIMRPhenomNSBH_delta2_prime(ft, fr)
        actual = float(delta2_prime_plus(torch.tensor(ft, dtype=torch.float64), torch.tensor(fr, dtype=torch.float64)))
        assert abs(actual - expected) < 1e-12


def test_gamma1_small_lambda_continuity():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    y = gamma1_prime_small_lambda(x)
    assert torch.allclose(y, torch.tensor([1.0, 1.25], dtype=torch.float64))
    (g,) = torch.autograd.grad(y.sum(), x)
    assert abs(float(g[1])) < 1e-12


def test_outer_pn_center_batch_dtype_and_gradient():
    base = torch.tensor([3019.9341093820635, 2534.1189767030205], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([0.9073742450220048, 0.9251287410598803], dtype=torch.float64, requires_grad=True)
    out = outer_pn_center(base, eps)
    assert out.dtype == torch.float64
    assert torch.allclose(out, base * eps)
    (gb, ge) = torch.autograd.grad(out.sum(), (base, eps))
    assert torch.all(torch.isfinite(gb)) and torch.all(torch.isfinite(ge))
    assert torch.allclose(gb, eps) and torch.allclose(ge, base)
