import math, torch
from torchwave.approximants.precession_nnlo_clean import raw_angles,effective_chip

def test_series_scaling_and_finite():
    v=torch.tensor([.1,.2,.4],dtype=torch.float64); a,e=raw_angles(v,torch.tensor(2/9,dtype=torch.float64),torch.tensor(1/3,dtype=torch.float64),torch.tensor(2/3,dtype=torch.float64),torch.tensor(.2,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64))
    assert torch.isfinite(a).all() and torch.isfinite(e).all()
    # Dominant alpha is v^-3, so low-v value is larger in magnitude.
    assert abs(a[0]) > abs(a[-1])

def test_effective_chip_exchange_and_aligned_zero():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    assert effective_chip(t(40),t(20),t(.3),t(.4),t(0),t(0)) > 0
    assert effective_chip(t(40),t(20),t(0),t(0),t(0),t(0)) == 0

def test_batched_parameters_and_frequency_grid_preserve_device_dtype():
    v = torch.tensor([.12, .2, .35], dtype=torch.float64)
    eta = torch.tensor([.20, .24], dtype=torch.float64)
    delta = torch.tensor([-.2, .1], dtype=torch.float64)
    m1 = torch.tensor([.55, .7], dtype=torch.float64)
    cl = torch.tensor([.1, -.2], dtype=torch.float64)
    cp = torch.tensor([.2, .3], dtype=torch.float64)
    a, e = raw_angles(v, eta, delta, m1, cl, cp)
    assert a.shape == (2, 3) and e.shape == (2, 3)
    assert a.dtype == torch.float64 and torch.isfinite(a).all()


def test_coefficients_are_autograd_differentiable_in_spin():
    chi = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(.24, dtype=torch.float64)
    aa, ee = __import__('torchwave.approximants.precession_nnlo_clean', fromlist=['coefficients']).coefficients(
        eta, torch.tensor(.1, dtype=torch.float64), torch.tensor(.7, dtype=torch.float64), chi,
        torch.tensor(.2, dtype=torch.float64))
    loss = aa.square().sum() + ee.square().sum()
    grad = torch.autograd.grad(loss, chi)[0]
    assert torch.isfinite(grad) and grad.abs() > 0
