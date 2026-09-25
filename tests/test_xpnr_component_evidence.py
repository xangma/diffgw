"""Independent scalar, Torch, and JAX evidence for printed XPNR components.

These tests qualify algebraic components only. They do not qualify the inherited
coefficient values, transition-spin attachment, tuned coprecessing carrier,
mode asymmetry, Euler-angle connection, or a complete XPNR waveform.
"""

import math

import numpy as np
import pytest
import torch

from diffgw_reference.xpnr_angles import (
    alpha_merger_ringdown,
    alpha_parameters,
    angle_calibration_window,
    beta_merger_ringdown,
    beta_parameters,
    deformed_carrier_coefficient,
    effective_ringdown_angular_frequency,
    evaluate_calibration_polynomial,
    freeze_fit_coordinates,
    higher_mode_frequency_map,
    single_spin_map_at_transition,
)
from torchwave.approximants.pnr_coefficients import COEFF_ARRAYS as TORCH_COEFFICIENTS
from torchwave.approximants.precession_pnr import (
    _eval_coeff_poly as torch_eval_coeff_poly,
    compute_pnr_alpha_coefficients as torch_alpha_coefficients,
    compute_pnr_beta_coefficients as torch_beta_coefficients,
    compute_pnr_single_spin as torch_single_spin,
    eval_pnr_alpha_mr as torch_alpha_mr,
    eval_pnr_beta_mr as torch_beta_mr,
    eval_pnr_linear_frequency_map as torch_frequency_map,
    pnr_angles_window as torch_angle_window,
)


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
from jaxwave.approximants.pnr_coefficients import COEFF_ARRAYS as JAX_COEFFICIENTS  # noqa: E402
from jaxwave.approximants.precession_pnr import (  # noqa: E402
    _eval_coeff_poly as jax_eval_coeff_poly,
    compute_pnr_alpha_coefficients as jax_alpha_coefficients,
    compute_pnr_beta_coefficients as jax_beta_coefficients,
    compute_pnr_single_spin as jax_single_spin,
    eval_pnr_alpha_mr as jax_alpha_mr,
    eval_pnr_beta_mr as jax_beta_mr,
    eval_pnr_linear_frequency_map as jax_frequency_map,
    pnr_angles_window as jax_angle_window,
)


POINTS = ((0.25, 0.2, -0.8), (0.16, 0.6, 0.3), (0.09, 0.79, 0.95))


@pytest.mark.parametrize("name", tuple(TORCH_COEFFICIENTS))
@pytest.mark.parametrize("eta,chi,costheta", POINTS)
def test_printed_polynomial_equation_has_scalar_torch_jax_agreement(name, eta, chi, costheta):
    """XO4a Eq. (35) agrees independently of the tensor backend."""
    coefficients = np.asarray(TORCH_COEFFICIENTS[name], dtype=np.float64)
    scalar = evaluate_calibration_polynomial(coefficients, eta, chi, costheta)
    torch_value = torch_eval_coeff_poly(
        torch.as_tensor(coefficients),
        torch.tensor(eta, dtype=torch.float64),
        torch.tensor(chi, dtype=torch.float64),
        torch.tensor(costheta, dtype=torch.float64),
    ).item()
    jax_value = float(
        jax_eval_coeff_poly(
            jnp.asarray(JAX_COEFFICIENTS[name], dtype=jnp.float64),
            jnp.asarray(eta, dtype=jnp.float64),
            jnp.asarray(chi, dtype=jnp.float64),
            jnp.asarray(costheta, dtype=jnp.float64),
        )
    )
    # Some inherited fits have severe cancellation at their boundary.  This is
    # still far below the precision of the stored decimal coefficients.
    assert torch_value == pytest.approx(scalar, rel=2e-10, abs=2e-10)
    assert jax_value == pytest.approx(scalar, rel=2e-10, abs=2e-10)


def test_inherited_tables_are_identical_between_backends_but_not_independently_sourced():
    """Duplication parity is useful engineering evidence, not coefficient provenance."""
    assert set(TORCH_COEFFICIENTS) == set(JAX_COEFFICIENTS)
    for name in TORCH_COEFFICIENTS:
        np.testing.assert_array_equal(
            np.asarray(TORCH_COEFFICIENTS[name], dtype=np.float64),
            np.asarray(JAX_COEFFICIENTS[name], dtype=np.float64),
        )


def test_transition_single_spin_map_matches_backend_algebra_when_given_same_spins():
    masses = (36.0, 24.0)
    spin1 = (0.2, -0.1, 0.3)
    spin2 = (-0.15, 0.05, -0.2)
    scalar = single_spin_map_at_transition(*masses, spin1, spin2)
    torch_values = torch_single_spin(*(torch.tensor(value, dtype=torch.float64) for value in (*masses, *spin1, *spin2)))
    jax_values = jax_single_spin(*(jnp.asarray(value) for value in (*masses, *spin1, *spin2)))
    for values in (torch_values, jax_values):
        assert float(values[0]) == pytest.approx(scalar.chi, rel=2e-14)
        assert float(values[1]) == pytest.approx(scalar.cos_theta_ls, rel=2e-14)
        assert float(values[3]) == pytest.approx(scalar.chi_effective, rel=2e-14)


def test_published_fit_transforms_match_scalar_reference_and_correct_b3_relation():
    q, chi, costheta, theta_final = 4.0, 0.6, 0.3, 0.4
    eta = q / (1.0 + q) ** 2
    fit = freeze_fit_coordinates(eta, chi, costheta, q)
    lambdas = {
        name: evaluate_calibration_polynomial(values, fit.eta, fit.chi, fit.cos_theta_ls)
        for name, values in TORCH_COEFFICIENTS.items()
    }
    scalar_alpha = alpha_parameters(
        fit.chi,
        fit.cos_theta_ls,
        *(lambdas[name] for name in ("A1", "A2", "A3", "A4")),
    )
    scalar_beta = beta_parameters(
        theta_final,
        fit.chi,
        fit.cos_theta_ls,
        *(lambdas[name] for name in ("B0", "B1", "B2", "B3", "B4", "B5")),
    )

    torch_args = tuple(torch.tensor(value, dtype=torch.float64) for value in (eta, chi, costheta))
    torch_alpha = torch_alpha_coefficients(*torch_args, torch.tensor(q), torch.device("cpu"), torch.float64)
    torch_beta = torch_beta_coefficients(
        *torch_args,
        torch.tensor(math.cos(theta_final), dtype=torch.float64),
        torch.tensor(q),
        torch.device("cpu"),
        torch.float64,
    )
    jax_alpha = jax_alpha_coefficients(*(jnp.asarray(value) for value in (eta, chi, costheta, q)))
    jax_beta = jax_beta_coefficients(*(jnp.asarray(value) for value in (eta, chi, costheta, math.cos(theta_final), q)))

    np.testing.assert_allclose([value.item() for value in torch_alpha], scalar_alpha, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose([value.item() for value in torch_beta], scalar_beta, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(np.asarray(jax_alpha), scalar_alpha, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(np.asarray(jax_beta), scalar_beta, rtol=2e-11, atol=2e-11)

    transverse_spin = fit.chi * math.sqrt(1.0 - fit.cos_theta_ls**2)
    expected_b3 = transverse_spin * (math.exp(lambdas["B3"]) - 200.0)
    obsolete_b3 = scalar_beta[2] * lambdas["B3"]
    assert scalar_beta[3] == pytest.approx(expected_b3, rel=2e-14)
    assert not math.isclose(scalar_beta[3], obsolete_b3, rel_tol=1e-3, abs_tol=1e-3)


def test_ansatz_values_and_frequency_derivatives_match_scalar_reference():
    mf = 0.08
    alpha_coefficients = (0.04, -0.02, 0.01, 0.12)
    beta_coefficients = (0.5, 0.03, -0.2, -0.08, 180.0, -0.03)
    step = 1.0e-7

    scalar_alpha = alpha_merger_ringdown(mf, *alpha_coefficients)
    scalar_beta = beta_merger_ringdown(mf, *beta_coefficients)
    fd_alpha = (
        alpha_merger_ringdown(mf + step, *alpha_coefficients) - alpha_merger_ringdown(mf - step, *alpha_coefficients)
    ) / (2.0 * step)
    fd_beta = (
        beta_merger_ringdown(mf + step, *beta_coefficients) - beta_merger_ringdown(mf - step, *beta_coefficients)
    ) / (2.0 * step)

    tmf = torch.tensor(mf, dtype=torch.float64, requires_grad=True)
    ta = torch_alpha_mr(tmf, *(torch.tensor(value, dtype=torch.float64) for value in alpha_coefficients))
    tb = torch_beta_mr(tmf, *(torch.tensor(value, dtype=torch.float64) for value in beta_coefficients))
    torch_da = torch.autograd.grad(ta, tmf, retain_graph=True)[0].item()
    torch_db = torch.autograd.grad(tb, tmf)[0].item()
    jax_da = float(jax.grad(lambda x: jax_alpha_mr(x, *alpha_coefficients))(mf))
    jax_db = float(jax.grad(lambda x: jax_beta_mr(x, *beta_coefficients))(mf))

    assert ta.item() == pytest.approx(scalar_alpha, rel=2e-12)
    assert tb.item() == pytest.approx(scalar_beta, rel=2e-12)
    assert float(jax_alpha_mr(mf, *alpha_coefficients)) == pytest.approx(scalar_alpha, rel=2e-12)
    assert float(jax_beta_mr(mf, *beta_coefficients)) == pytest.approx(scalar_beta, rel=2e-12)
    assert torch_da == pytest.approx(fd_alpha, rel=2e-8, abs=2e-8)
    assert torch_db == pytest.approx(fd_beta, rel=2e-8, abs=2e-8)
    assert jax_da == pytest.approx(fd_alpha, rel=2e-8, abs=2e-8)
    assert jax_db == pytest.approx(fd_beta, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize("q,chi", ((8.5, 0.85), (10.25, 1.025), (12.0, 0.7), (4.0, 1.2)))
def test_angle_window_has_scalar_torch_jax_agreement(q, chi):
    """Backends agree on the documented endpoint-completing interpretation."""
    scalar = angle_calibration_window(q, chi)
    torch_value = torch_angle_window(torch.tensor(q), torch.tensor(chi)).item()
    jax_value = float(jax_angle_window(jnp.asarray(q), jnp.asarray(chi)))
    assert torch_value == pytest.approx(scalar, abs=2e-7)
    assert jax_value == pytest.approx(scalar, abs=2e-14)


@pytest.mark.parametrize("mf", (0.02, 0.04, 0.07, 0.1))
def test_higher_mode_map_has_scalar_torch_jax_agreement_and_continuity(mf):
    arguments = (3, 0.04, 0.07, 0.08, 0.11)
    scalar = higher_mode_frequency_map(mf, *arguments)
    tensor_arguments = tuple(torch.tensor(value, dtype=torch.float64) for value in arguments[1:])
    torch_value = torch_frequency_map(torch.tensor(mf), 3, 3, *tensor_arguments).item()
    jax_value = float(jax_frequency_map(jnp.asarray(mf), 3, 3, *(jnp.asarray(v) for v in arguments[1:])))
    assert torch_value == pytest.approx(scalar, abs=2e-8)
    assert jax_value == pytest.approx(scalar, abs=2e-14)


def test_printed_carrier_relations_and_nonprecessing_component_reductions():
    transverse = 0.6 * math.sqrt(1.0 - 0.5**2)
    assert deformed_carrier_coefficient(1.25, 0.6, 0.5, 4.0) == pytest.approx(1.25 + transverse * 4.0)
    assert effective_ringdown_angular_frequency(0.5, 3, 0.5, 0.45, 0.4) == pytest.approx(0.5 - 3.0 * 0.5 * (0.45 - 0.4))

    alpha = alpha_parameters(0.7, 1.0, 2.0, -3.0, 0.4, 0.5)
    beta = beta_parameters(0.3, 0.7, 1.0, 2.0, -3.0, 0.4, 0.5, 10.0, -0.2)
    assert alpha[:2] == (0.0, 0.0)
    assert beta[:4] == (0.3, 0.0, 0.0, 0.0)
    assert deformed_carrier_coefficient(1.25, 0.7, 1.0, 4.0) == 1.25
    assert effective_ringdown_angular_frequency(0.5, 3, 1.0, 0.45, 0.4) == 0.5
