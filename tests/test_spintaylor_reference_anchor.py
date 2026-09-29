"""Reference-frequency gauge checks for the native SpinTaylor angle routes."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jaxwave.approximants.precession_spintaylor import evaluate_spintaylor_precession as jax_angles
from torchwave.approximants.precession_spintaylor import evaluate_spintaylor_precession as torch_angles


PARAMS = dict(
    f_ref=37.3,
    mass1=36.0,
    mass2=29.0,
    spin1x=0.3,
    spin1y=0.2,
    spin1z=-0.1,
    spin2x=0.1,
    spin2y=-0.2,
    spin2z=0.2,
    inclination=0.6,
    phi_ref=0.3,
)


@pytest.mark.parametrize("convention", [0, 1])
def test_reference_angle_gauge_uses_interpolated_node(convention):
    # f_ref lies strictly between integration nodes on the fixed 15 Hz grid.
    f = [20.0, PARAMS["f_ref"], 80.0]
    ta, tb, tg, *tframe = torch_angles(
        f=torch.tensor(f, dtype=torch.float64), convention=convention, **PARAMS
    )
    ja, jb, jg, *jframe = jax_angles(
        f=jnp.asarray(f, dtype=jnp.float64), convention=convention, **PARAMS
    )

    assert abs(float(ta[1])) < 1e-12
    assert abs(float(ja[1])) < 1e-12
    # Independent scalar J0/L0 geometry gives theta_J = 0.12616479751 rad.
    # This also checks that the reference state was evolved to the lower anchor.
    assert abs(float(tb[1]) - 0.12616479751061854) < 3e-6
    assert abs(float(jb[1]) - 0.12616479751061854) < 3e-6
    if convention == 0:
        assert abs(float(tg[1])) < 1e-12
        assert abs(float(jg[1])) < 1e-12
    for torch_value, jax_value in zip((ta, tb, tg, *tframe), (ja, jb, jg, *jframe)):
        np.testing.assert_allclose(
            torch_value.detach().cpu().numpy(), np.asarray(jax_value), rtol=1e-9, atol=1e-10
        )


def test_reference_anchor_preserves_spin_gradients():
    f_t = torch.tensor([20.0, 37.3, 80.0], dtype=torch.float64)
    spin_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    tparams = dict(PARAMS, spin1x=spin_t)
    alpha_t = torch_angles(f=f_t, convention=0, **tparams)[0]
    grad_t = torch.autograd.grad(alpha_t[-1], spin_t)[0]

    f_j = jnp.asarray([20.0, 37.3, 80.0], dtype=jnp.float64)
    grad_j = jax.grad(
        lambda spin: jax_angles(f=f_j, convention=0, **dict(PARAMS, spin1x=spin))[0][-1]
    )(jnp.asarray(0.3, dtype=jnp.float64))

    assert torch.isfinite(grad_t)
    assert bool(jnp.isfinite(grad_j))
    np.testing.assert_allclose(float(grad_t), float(grad_j), rtol=1e-7, atol=1e-8)

    jit_angles = jax.jit(lambda f: jax_angles(f=f, convention=0, **PARAMS)[:3])(f_j)
    assert abs(float(jit_angles[0][1])) < 1e-12
    assert abs(float(jit_angles[2][1])) < 1e-12


@pytest.mark.parametrize("convention", [0, 1])
def test_shared_frequency_angles_do_not_depend_on_output_grid(convention):
    grids = ([37.3, 80.0], [20.0, 37.3, 80.0])
    for angles, asarray in (
        (torch_angles, lambda x: torch.tensor(x, dtype=torch.float64)),
        (jax_angles, lambda x: jnp.asarray(x, dtype=jnp.float64)),
    ):
        narrow = angles(f=asarray(grids[0]), convention=convention, **PARAMS)
        wide = angles(f=asarray(grids[1]), convention=convention, **PARAMS)
        for narrow_angle, wide_angle in zip(narrow[:3], wide[:3]):
            np.testing.assert_allclose(
                np.asarray(narrow_angle)[-1], np.asarray(wide_angle)[-1], rtol=0.0, atol=1e-10
            )
