"""Contracts for independent PhenomT trajectory and angle references."""

import numpy as np
import pytest

pytest.importorskip("mpmath")
pytest.importorskip("scipy")

from diffgw_reference.phenomt_angle_continuation import (
    apply_angle_option,
    meco_value_and_slope,
    minimal_rotation_gamma,
)
from diffgw_reference.phenomt_numerical_trajectory import (
    euler_angles,
    inertial_basis,
    integrate_trajectory,
    rotation_matrix,
)
from diffgw_reference.phenomt_spin_evolution import evaluate_spin_derivatives


def test_scalar_rhs_accepts_binary64_ode_state_but_rejects_material_drift():
    lhat = np.array([2e-7, -3e-7, 1.0])
    lhat /= np.linalg.norm(lhat)
    lhat[2] += 2e-13
    evaluate_spin_derivatives(0.2, lhat, (0.1, 0.0, 0.2), (0.0, 0.01, 0.0), 0.9, 0.1)
    with pytest.raises(ValueError, match="unit vector"):
        evaluate_spin_derivatives(0.2, (0.0, 0.0, 1.00001), (0.1, 0.0, 0.2),
                                  (0.0, 0.01, 0.0), 0.9, 0.1)


def test_independent_dop853_event_transport_and_angles_preserve_triad():
    def velocity(time):
        return 0.2 + 1e-3 * (time + 10.0)
    trajectory = integrate_trajectory(
        velocity,
        0.2,
        (-11.0, -9.0),
        0.0,
        10 / 11,
        1 / 11,
        (0.2, 0.1, 0.8),
        (0.1, 0.0, 0.1),
        rtol=3e-11,
        atol=3e-13,
        dps=40,
    )
    assert trajectory.reference_time == pytest.approx(-10.0, abs=2e-13)
    times = np.linspace(trajectory.reference_time, 0.0, 17)
    states = trajectory.evaluate(times)
    lhat, spin1, spin2, tangent = np.split(states, 4, axis=-1)
    assert np.max(abs(np.linalg.norm(lhat, axis=-1) - 1)) < 2e-11
    assert np.max(abs(np.linalg.norm(spin1, axis=-1) - np.linalg.norm(spin1[0]))) < 2e-11
    assert np.max(abs(np.linalg.norm(spin2, axis=-1) - np.linalg.norm(spin2[0]))) < 2e-11
    assert np.max(abs(np.sum(lhat * tangent, axis=-1))) < 2e-11

    rhs = evaluate_spin_derivatives(0.2, lhat[0], spin1[0], spin2[0], 10 / 11, 1 / 11,
                                    include_n3lo_ss=False)
    basis = inertial_basis(trajectory.initial, float(rhs.orbital_angular_momentum))
    rotations = rotation_matrix(*euler_angles(states, basis))
    jhat, ex, ey = basis
    lhat_components = np.stack([lhat @ vector for vector in (ex, ey, jhat)], axis=-1)
    tangent_components = np.stack([tangent @ vector for vector in (ex, ey, jhat)], axis=-1)
    np.testing.assert_allclose(rotations[..., 2], lhat_components, rtol=0, atol=3e-11)
    np.testing.assert_allclose(rotations[..., 0], tangent_components, rtol=0, atol=3e-11)


@pytest.mark.parametrize("supply_table_derivative", [False, True])
def test_adaptive_minimal_rotation_matches_closed_form(supply_table_derivative):
    def alpha(time):
        return 0.4 * time**2 + 0.3 * time - 0.2
    beta = 0.47
    reference_time = -1.3
    gamma_reference = 0.81
    times = np.asarray([-1.3, -0.8, -0.1, 0.0])
    options = {}
    if supply_table_derivative:
        options.update(
            alpha_derivative=lambda time: 0.8 * time + 0.3,
            integration_breakpoints=np.linspace(reference_time, 0.0, 17),
        )
    actual = minimal_rotation_gamma(
        times, lambda time: (alpha(time), beta), reference_time, gamma_reference,
        **options,
    )
    expected = gamma_reference - np.cos(beta) * (alpha(times) - alpha(reference_time))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-11)


def test_scalar_meco_and_ringdown_continuations_are_continuous():
    def raw(time):
        return 0.4 * time**2 + 0.3 * time, -0.2 * time**2 + 0.1, 0.7 + 0.2 * time

    meco_time = -0.2
    value, slope = meco_value_and_slope(raw, meco_time)
    np.testing.assert_allclose(value, raw(meco_time)[:2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(slope, (0.8 * meco_time + 0.3, -0.4 * meco_time), rtol=0, atol=2e-11)
    def derivative(time):
        return 0.8 * time + 0.3, -0.4 * time, 0.2
    _, supplied_slope = meco_value_and_slope(raw, meco_time, derivative)
    np.testing.assert_allclose(supplied_slope, slope, rtol=0, atol=2e-11)

    epsilon = 1e-9
    times = np.asarray([meco_time - epsilon, meco_time, -epsilon, 0.0, epsilon])
    alpha, beta, gamma = apply_angle_option(
        times,
        raw,
        22321,
        reference_time=-0.8,
        meco_time=meco_time,
        ringdown_alpha_slope=1.7,
        raw_derivatives=derivative,
        integration_breakpoints=np.linspace(-0.8, 0.0, 17),
    )
    assert abs(alpha[1] - alpha[0]) < 1e-8
    assert abs(beta[1] - beta[0]) < 1e-8
    assert abs(alpha[-1] - alpha[-2] - 1.7 * epsilon) < 1e-13
    assert gamma[-1] == pytest.approx(gamma[-2] - np.cos(beta[-2]) * 1.7 * epsilon, abs=2e-13)
