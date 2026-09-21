import torch

from torchwave.approximants._phenomt_angle_options import (
    apply_angle_option,
    linear_meco_continuation,
    minimal_rotation_gamma,
)


def _angles(t):
    return 0.4 * t + 0.1 * t.square(), 0.2 + 0.05 * t


def test_minimal_rotation_is_grid_independent_and_differentiable():
    ref = torch.tensor(-0.2, dtype=torch.float64)
    gamma_ref = torch.tensor(0.3, dtype=torch.float64)
    coarse = torch.tensor([-0.1, 0.0, 0.02], dtype=torch.float64)
    dense = torch.linspace(-0.1, 0.02, 19, dtype=torch.float64)
    coarse.requires_grad_()
    gc = minimal_rotation_gamma(coarse, _angles, gamma_ref, reference_time=ref, n_grid=256)
    gd = minimal_rotation_gamma(dense, _angles, gamma_ref, reference_time=ref, n_grid=256)
    torch.testing.assert_close(gc, gd[[0, 15, 18]], rtol=0, atol=2e-6)
    gc.sum().backward()
    assert torch.isfinite(coarse.grad).all()


def test_meco_continuation_matches_value_and_slope_at_meco():
    t = torch.linspace(-0.2, 0.05, 101, dtype=torch.float64)
    meco = torch.tensor(-0.1, dtype=torch.float64)
    a, b = _angles(t)
    aa, bb = linear_meco_continuation(t, a, b, meco, angle_evaluator=_angles)
    i = torch.argmin((t - meco).abs())
    torch.testing.assert_close(aa[i], a[i], atol=2e-4, rtol=0)
    torch.testing.assert_close(bb[i], b[i], atol=2e-4, rtol=0)


def test_analytical_whole_waveform_option_does_not_attach_ringdown():
    t = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64)
    a, b = _angles(t)
    g = 0.3 * t
    out = apply_angle_option(t, a, b, g, a[-1], b[-1], g[-1], torch.tensor(2.0), 10200)
    torch.testing.assert_close(out[0], a)
    torch.testing.assert_close(out[1], b)
    torch.testing.assert_close(out[2], g)


def test_option_21_gamma_integrates_continued_angles_in_both_backends():
    import numpy as np
    import jax.numpy as jnp
    from jaxwave.approximants._phenomt_angle_options import apply_angle_option as apply_jax

    for xp, apply in ((torch, apply_angle_option), (jnp, apply_jax)):
        dtype = xp.float64
        array = xp.as_tensor if xp is torch else xp.asarray
        t = array([-0.1, -0.05, -0.01, 0.0, 0.02], dtype=dtype)
        ref = array(-0.1, dtype=dtype)
        gref = array(0.3, dtype=dtype)
        slope = array(2.0, dtype=dtype)
        def raw(q):
            return 0.4*q+0.1*q*q, 0.2+0.05*q
        a, b = raw(t)
        a0, b0 = raw(t*0)
        actual = apply(t, a, b, t*0, a0[0], b0[0], gref, slope, 10221,
                       angle_evaluator=raw, meco_time=ref, reference_time=ref,
                       gamma_reference=gref)
        # After MECO alpha_dot is the constant 0.38; the raw quadratic
        # alpha would yield a different gamma and fail this integral check.
        clipped = xp.minimum(t, t*0)
        expected = gref - 0.38/0.05*(xp.sin(0.2+0.05*clipped)-xp.sin(0.2+0.05*ref))
        expected = expected-xp.cos(b0[0])*slope*xp.maximum(t, t*0)
        np.testing.assert_allclose(np.asarray(actual[2]), np.asarray(expected), rtol=0, atol=1e-9)
