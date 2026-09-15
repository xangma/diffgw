import torch

from torchwave.approximants.precession_beta_clean import beta, cos_beta, orbital_angular_momentum


def test_beta_is_aligned_limit_and_physical():
    args=(40.,20.,[20.,50.,100.],.0,.0,.4,.0,.0,-.2)
    b=beta(*args)
    torch.testing.assert_close(b, torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.isfinite(b).all()


def test_batch_and_autograd():
    f=torch.tensor([25.,50.,100.],dtype=torch.float64,requires_grad=True)
    b=beta(35.,25.,f,.2,.1,.2,-.1,.15,-.1)
    g=torch.autograd.grad(b.sum(),f)[0]
    assert b.shape==(3,) and g.shape==(3,)
    assert torch.isfinite(g).all()


def test_eq17_is_even_under_inplane_spin_reversal():
    kw=(35.,25.,80.,.2,.1,.2,-.1,.15,-.1)
    rev=(35.,25.,80.,-.2,-.1,.2,.1,-.15,-.1)
    torch.testing.assert_close(cos_beta(*kw),cos_beta(*rev),atol=1e-14,rtol=0)


def test_eq26_nonspin_limit_is_positive_and_scales_with_frequency():
    L=orbital_angular_momentum(35.,25.,torch.tensor([20.,100.],dtype=torch.float64),0.,0.)
    assert torch.all(L>0) and L[0]>L[1]


def test_python_scalar_masses_are_float64():
    # Decimal Python scalars must not inherit the process default dtype.
    L = orbital_angular_momentum(
        35.125, 24.875, torch.tensor([20.0, 80.0], dtype=torch.float64), 0.2, -0.1
    )
    assert L.dtype == torch.float64
    assert L.shape == (2,) and torch.isfinite(L).all()
