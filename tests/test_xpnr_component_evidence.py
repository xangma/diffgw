"""Independent scalar, Torch, and JAX evidence for printed XPNR components.

These tests qualify algebraic components only. They do not qualify the inherited
coefficient values, transition-spin attachment, tuned coprecessing carrier,
mode asymmetry, Euler-angle connection, or a complete XPNR waveform.
"""

import math
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

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
    single_spin_map_from_transition_vectors,
)

POINTS = ((0.25, 0.2, -0.8), (0.16, 0.6, 0.3), (0.09, 0.79, 0.95))
COEFFICIENT_NAMES = ("A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3", "B4", "B5", "Bf")


@pytest.fixture(scope="module")
def backends():
    """Load each installed frontend without making either one a test dependency."""
    available = {}
    if find_spec("torch") is not None:
        import torch
        from torchwave.approximants.pnr_coefficients import COEFF_ARRAYS
        from torchwave.approximants.precession_pnr import (
            _eval_coeff_poly,
            compute_pnr_alpha_coefficients,
            compute_pnr_beta_coefficients,
            compute_pnr_single_spin,
            eval_pnr_alpha_mr,
            eval_pnr_beta_mr,
            eval_pnr_linear_frequency_map,
            pnr_angles_window,
        )

        available["torch"] = SimpleNamespace(
            module=torch,
            coefficients=COEFF_ARRAYS,
            eval_coeff_poly=_eval_coeff_poly,
            alpha_coefficients=compute_pnr_alpha_coefficients,
            beta_coefficients=compute_pnr_beta_coefficients,
            single_spin=compute_pnr_single_spin,
            alpha_mr=eval_pnr_alpha_mr,
            beta_mr=eval_pnr_beta_mr,
            frequency_map=eval_pnr_linear_frequency_map,
            angle_window=pnr_angles_window,
        )
    if find_spec("jax") is not None:
        import jax
        import jax.numpy as jnp
        from jaxwave.approximants.pnr_coefficients import COEFF_ARRAYS
        from jaxwave.approximants.precession_pnr import (
            _eval_coeff_poly,
            compute_pnr_alpha_coefficients,
            compute_pnr_beta_coefficients,
            compute_pnr_single_spin,
            eval_pnr_alpha_mr,
            eval_pnr_beta_mr,
            eval_pnr_linear_frequency_map,
            pnr_angles_window,
        )

        jax.config.update("jax_enable_x64", True)
        available["jax"] = SimpleNamespace(
            module=jax,
            arrays=jnp,
            coefficients=COEFF_ARRAYS,
            eval_coeff_poly=_eval_coeff_poly,
            alpha_coefficients=compute_pnr_alpha_coefficients,
            beta_coefficients=compute_pnr_beta_coefficients,
            single_spin=compute_pnr_single_spin,
            alpha_mr=eval_pnr_alpha_mr,
            beta_mr=eval_pnr_beta_mr,
            frequency_map=eval_pnr_linear_frequency_map,
            angle_window=pnr_angles_window,
        )
    return available


def require_frontend(backends):
    if not backends:
        pytest.skip("component-to-frontend checks require Torch or JAX")
    return next(iter(backends.values()))


@pytest.mark.parametrize("coefficient_name", COEFFICIENT_NAMES)
@pytest.mark.parametrize("eta,chi,costheta", POINTS)
def test_printed_polynomial_equation_has_scalar_torch_jax_agreement(backends, coefficient_name, eta, chi, costheta):
    """XO4a Eq. (35) agrees independently of the tensor backend."""
    first_backend = require_frontend(backends)
    coefficients = np.asarray(first_backend.coefficients[coefficient_name], dtype=np.float64)
    scalar = evaluate_calibration_polynomial(coefficients, eta, chi, costheta)
    for backend_name, backend in backends.items():
        if backend_name == "torch":
            module = backend.module
            value = backend.eval_coeff_poly(
                module.as_tensor(coefficients),
                module.tensor(eta, dtype=module.float64),
                module.tensor(chi, dtype=module.float64),
                module.tensor(costheta, dtype=module.float64),
            ).item()
        else:
            arrays = backend.arrays
            value = float(
                backend.eval_coeff_poly(
                    arrays.asarray(backend.coefficients[coefficient_name], dtype=arrays.float64),
                    arrays.asarray(eta, dtype=arrays.float64),
                    arrays.asarray(chi, dtype=arrays.float64),
                    arrays.asarray(costheta, dtype=arrays.float64),
                )
            )
        # Some inherited fits have severe cancellation at their boundary.  This
        # is still far below the precision of the stored decimal coefficients.
        assert value == pytest.approx(scalar, rel=2e-10, abs=2e-10)


def test_inherited_tables_are_identical_between_backends_but_not_independently_sourced(backends):
    """Duplication parity is useful engineering evidence, not coefficient provenance."""
    if set(backends) != {"torch", "jax"}:
        pytest.skip("coefficient-table parity requires both frontends")
    torch_coefficients = backends["torch"].coefficients
    jax_coefficients = backends["jax"].coefficients
    assert set(torch_coefficients) == set(jax_coefficients)
    for name in torch_coefficients:
        np.testing.assert_array_equal(
            np.asarray(torch_coefficients[name], dtype=np.float64),
            np.asarray(jax_coefficients[name], dtype=np.float64),
        )


def test_transition_single_spin_map_matches_backend_algebra_when_given_same_spins(backends):
    masses = (36.0, 24.0)
    spin1 = (0.2, -0.1, 0.3)
    spin2 = (-0.15, 0.05, -0.2)
    scalar = single_spin_map_at_transition(*masses, spin1, spin2)
    require_frontend(backends)
    for name, backend in backends.items():
        if name == "torch":
            module = backend.module
            values = backend.single_spin(
                *(module.tensor(value, dtype=module.float64) for value in (*masses, *spin1, *spin2))
            )
        else:
            values = backend.single_spin(*(backend.arrays.asarray(value) for value in (*masses, *spin1, *spin2)))
        assert float(values[0]) == pytest.approx(scalar.chi, rel=2e-14)
        assert float(values[1]) == pytest.approx(scalar.cos_theta_ls, rel=2e-14)
        assert float(values[3]) == pytest.approx(scalar.chi_effective, rel=2e-14)


def test_transition_vector_map_uses_instantaneous_orbital_plane_and_is_rotation_invariant():
    masses = (36.0, 24.0)
    spin1 = (0.2, -0.1, 0.3)
    spin2 = (-0.15, 0.05, -0.2)

    z_aligned = single_spin_map_from_transition_vectors(*masses, spin1, spin2, (0.0, 0.0, 2.0))
    component_map = single_spin_map_at_transition(*masses, spin1, spin2)
    assert z_aligned == component_map

    orbital_direction = (1.0, 2.0, 2.0)
    mapped = single_spin_map_from_transition_vectors(*masses, spin1, spin2, orbital_direction)
    rotated = single_spin_map_from_transition_vectors(
        *masses,
        (spin1[2], spin1[0], spin1[1]),
        (spin2[2], spin2[0], spin2[1]),
        (orbital_direction[2], orbital_direction[0], orbital_direction[1]),
    )
    np.testing.assert_allclose(
        tuple(rotated.__dict__.values()),
        tuple(mapped.__dict__.values()),
        rtol=2e-15,
        atol=2e-15,
    )

    lhat = np.asarray(orbital_direction) / np.linalg.norm(orbital_direction)
    parallel1 = float(np.dot(spin1, lhat))
    parallel2 = float(np.dot(spin2, lhat))
    transverse_sum = masses[0] ** 2 * (np.asarray(spin1) - parallel1 * lhat) + masses[1] ** 2 * (
        np.asarray(spin2) - parallel2 * lhat
    )
    assert mapped.chi_perpendicular == pytest.approx(np.linalg.norm(transverse_sum) / masses[0] ** 2)


def test_published_fit_transforms_match_scalar_reference_and_correct_b3_relation(backends):
    q, chi, costheta, theta_final = 4.0, 0.6, 0.3, 0.4
    eta = q / (1.0 + q) ** 2
    fit = freeze_fit_coordinates(eta, chi, costheta, q)
    lambdas = {
        name: evaluate_calibration_polynomial(values, fit.eta, fit.chi, fit.cos_theta_ls)
        for name, values in require_frontend(backends).coefficients.items()
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

    for name, backend in backends.items():
        if name == "torch":
            module = backend.module
            args = tuple(module.tensor(value, dtype=module.float64) for value in (eta, chi, costheta))
            actual_alpha = backend.alpha_coefficients(*args, module.tensor(q), module.device("cpu"), module.float64)
            actual_beta = backend.beta_coefficients(
                *args,
                module.tensor(math.cos(theta_final), dtype=module.float64),
                module.tensor(q),
                module.device("cpu"),
                module.float64,
            )
            actual_alpha = [value.item() for value in actual_alpha]
            actual_beta = [value.item() for value in actual_beta]
        else:
            arrays = backend.arrays
            actual_alpha = backend.alpha_coefficients(*(arrays.asarray(value) for value in (eta, chi, costheta, q)))
            actual_beta = backend.beta_coefficients(
                *(arrays.asarray(value) for value in (eta, chi, costheta, math.cos(theta_final), q))
            )
        np.testing.assert_allclose(np.asarray(actual_alpha), scalar_alpha, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(np.asarray(actual_beta), scalar_beta, rtol=2e-11, atol=2e-11)

    transverse_spin = fit.chi * math.sqrt(1.0 - fit.cos_theta_ls**2)
    expected_b3 = transverse_spin * (math.exp(lambdas["B3"]) - 200.0)
    obsolete_b3 = scalar_beta[2] * lambdas["B3"]
    assert scalar_beta[3] == pytest.approx(expected_b3, rel=2e-14)
    assert not math.isclose(scalar_beta[3], obsolete_b3, rel_tol=1e-3, abs_tol=1e-3)


def test_ansatz_values_and_frequency_derivatives_match_scalar_reference(backends):
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

    require_frontend(backends)
    for name, backend in backends.items():
        if name == "torch":
            module = backend.module
            frequency = module.tensor(mf, dtype=module.float64, requires_grad=True)
            alpha = backend.alpha_mr(
                frequency, *(module.tensor(value, dtype=module.float64) for value in alpha_coefficients)
            )
            beta = backend.beta_mr(
                frequency, *(module.tensor(value, dtype=module.float64) for value in beta_coefficients)
            )
            derivative_alpha = module.autograd.grad(alpha, frequency, retain_graph=True)[0].item()
            derivative_beta = module.autograd.grad(beta, frequency)[0].item()
            alpha = alpha.detach().item()
            beta = beta.detach().item()
        else:
            module = backend.module
            alpha = backend.alpha_mr(mf, *alpha_coefficients)
            beta = backend.beta_mr(mf, *beta_coefficients)
            derivative_alpha = float(module.grad(lambda x: backend.alpha_mr(x, *alpha_coefficients))(mf))
            derivative_beta = float(module.grad(lambda x: backend.beta_mr(x, *beta_coefficients))(mf))
        assert float(alpha) == pytest.approx(scalar_alpha, rel=2e-12)
        assert float(beta) == pytest.approx(scalar_beta, rel=2e-12)
        assert derivative_alpha == pytest.approx(fd_alpha, rel=2e-8, abs=2e-8)
        assert derivative_beta == pytest.approx(fd_beta, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize("q,chi", ((8.5, 0.85), (10.25, 1.025), (12.0, 0.7), (4.0, 1.2)))
def test_angle_window_has_scalar_torch_jax_agreement(backends, q, chi):
    """Backends agree on the documented endpoint-completing interpretation."""
    scalar = angle_calibration_window(q, chi)
    require_frontend(backends)
    for name, backend in backends.items():
        if name == "torch":
            value = backend.angle_window(backend.module.tensor(q), backend.module.tensor(chi)).item()
            tolerance = 2e-7
        else:
            value = float(backend.angle_window(backend.arrays.asarray(q), backend.arrays.asarray(chi)))
            tolerance = 2e-14
        assert value == pytest.approx(scalar, abs=tolerance)


@pytest.mark.parametrize("mf", (0.02, 0.04, 0.07, 0.1))
def test_higher_mode_map_has_scalar_torch_jax_agreement_and_continuity(backends, mf):
    arguments = (3, 0.04, 0.07, 0.08, 0.11)
    scalar = higher_mode_frequency_map(mf, *arguments)
    require_frontend(backends)
    for name, backend in backends.items():
        if name == "torch":
            module = backend.module
            tensor_arguments = tuple(module.tensor(value, dtype=module.float64) for value in arguments[1:])
            value = backend.frequency_map(module.tensor(mf), 3, 3, *tensor_arguments).item()
            tolerance = 2e-8
        else:
            arrays = backend.arrays
            value = float(backend.frequency_map(arrays.asarray(mf), 3, 3, *(arrays.asarray(v) for v in arguments[1:])))
            tolerance = 2e-14
        assert value == pytest.approx(scalar, abs=tolerance)


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
