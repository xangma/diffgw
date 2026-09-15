import pytest
import torch
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from torchwave.approximants.kerr_qnm_table_convention import (
    KNOTS, rounded_knot_values, signed_qnm_220_table_convention,
)


def test_knots_are_regenerated_and_exact_after_rounding():
    x = torch.tensor(KNOTS, dtype=torch.float64)
    got = torch.stack(signed_qnm_220_table_convention(x), dim=-1)
    assert torch.equal(got, rounded_knot_values().T)


def test_independent_validation_spins_and_shapes():
    x = torch.tensor([-.9231, -.4732, -.0213, .2841, .5763, .8123, .9641, .9971], dtype=torch.float64)
    got = signed_qnm_220_table_convention(x)
    ref = signed_qnm_220(x)
    assert max((got[i] - ref[i]).abs().max().item() for i in range(2)) < 1e-3
    z = signed_qnm_220_table_convention(torch.tensor(0.0, dtype=torch.float64))
    assert all(v.ndim == 0 and torch.isfinite(v) for v in z)


def test_derivative_and_mixed_batch():
    x = torch.tensor([-.4, 0.0, .7], dtype=torch.float64, requires_grad=True)
    y = signed_qnm_220_table_convention(x)[0].sum()
    (g,) = torch.autograd.grad(y, x)
    h = 1e-6
    xp = x.detach() + h; xm = x.detach() - h
    fd = (signed_qnm_220_table_convention(xp)[0] - signed_qnm_220_table_convention(xm)[0])/(2*h)
    assert torch.allclose(g, fd, rtol=2e-5, atol=2e-7)


def test_matches_independent_scipy_natural_spline():
    scipy = pytest.importorskip("scipy.interpolate")
    x = torch.tensor([-.9231, -.4732, -.0213, .2841, .5763, .8123, .9641], dtype=torch.float64)
    knots = torch.tensor(KNOTS, dtype=torch.float64).numpy()
    values = rounded_knot_values().numpy()
    for channel in range(2):
        spline = scipy.CubicSpline(knots, values[channel], bc_type="natural")
        got = signed_qnm_220_table_convention(x)[channel].detach().numpy()
        assert (abs(got - spline(x.numpy())) < 2e-12).all()
        assert (abs((signed_qnm_220_table_convention(x + 1e-6)[channel].detach().numpy()
                        - signed_qnm_220_table_convention(x - 1e-6)[channel].detach().numpy())
                    / 2e-6 - spline(x.numpy(), 1)) < 2e-5).all()


def test_domain():
    signed_qnm_220_table_convention(torch.tensor([-.9996, .9996]))
    with pytest.raises(ValueError):
        signed_qnm_220_table_convention(torch.tensor(.99961))
