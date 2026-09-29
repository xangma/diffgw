"""Reference-frequency geometry and output-grid invariance for SpinTaylor.

The reference constants come from scripts/qualification/spintaylor_reference_geometry.py,
which evaluates Pratten et al. (2021), Appendix C, with mpmath at 60 digits.
"""

import numpy as np
import pytest
import torch

from torchwave.approximants.precession_spintaylor import evaluate_spintaylor_precession as torch_angles


PARAMS = dict(
    f_ref=37.3, mass1=36.0, mass2=29.0,
    spin1x=0.3, spin1y=0.2, spin1z=-0.1,
    spin2x=0.1, spin2y=-0.2, spin2z=0.2,
    inclination=0.6, phi_ref=0.0,
)
REFERENCE = (1.5743964450330335, 0.12616479751384346, 2.951486722303569)


def _check_reference_and_grid(fn, array):
    narrow = fn(f=array([37.3, 80.0]), **PARAMS)
    wide = fn(f=array([20.0, 37.3, 80.0]), **PARAMS)
    for index, expected in enumerate(REFERENCE):
        np.testing.assert_allclose(np.asarray(narrow[index])[0], expected, rtol=0, atol=2e-11)
        np.testing.assert_allclose(np.asarray(wide[index])[1], expected, rtol=0, atol=2e-11)
        np.testing.assert_allclose(np.asarray(narrow[index])[-1], np.asarray(wide[index])[-1], rtol=0, atol=2e-11)
    assert np.asarray(wide[1])[0] < np.asarray(wide[1])[1]  # backward-evolved L at 20 Hz


def test_torch_reference_and_grid_invariance():
    _check_reference_and_grid(torch_angles, lambda x: torch.tensor(x, dtype=torch.float64))


def test_torch_reference_angle_gradient():
    spin1x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    params = {**PARAMS, "spin1x": spin1x}
    alpha, beta, gamma, *_ = torch_angles(f=torch.tensor([20.0, 37.3, 80.0], dtype=torch.float64), **params)
    gradients = torch.autograd.grad(alpha[1] + beta[0] + gamma[2], spin1x)[0]
    assert torch.isfinite(gradients) and gradients.abs() > 1e-8


def test_jax_reference_grid_and_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jaxwave.approximants.precession_spintaylor import evaluate_spintaylor_precession as jax_angles

    assert jax.config.jax_enable_x64
    _check_reference_and_grid(jax_angles, lambda x: jnp.asarray(x, dtype=jnp.float64))

    def scalar(spin1x):
        params = {**PARAMS, "spin1x": spin1x}
        alpha, beta, gamma, *_ = jax_angles(f=jnp.asarray([20.0, 37.3, 80.0]), **params)
        return alpha[1] + beta[0] + gamma[2]

    gradient = jax.grad(scalar)(jnp.asarray(0.3, dtype=jnp.float64))
    assert np.isfinite(gradient) and abs(float(gradient)) > 1e-8
