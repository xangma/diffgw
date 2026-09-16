"""Contract tests for the clean TEOB auxiliary-grid NQC assembly."""

import pytest
import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    TEOBMergerNQCError,
    TEOBMergerNQCTargets,
    assemble_teob_merger_nqc,
    matching_index,
    phase_from_mode,
    published_2017_nqc_targets,
    quartic_derivative,
)


def _synthetic(dtype=torch.float64):
    """Return a scalar series with a resolved interior frequency maximum."""

    n = 81
    step = torch.as_tensor(0.5, dtype=dtype)
    time = torch.arange(n, dtype=dtype) * step
    radius = 20.0 - 0.08 * time
    state = torch.stack((
        radius,
        0.01 * time.square(),
        -0.04 - 0.001 * time,
        3.0 + 0.01 * time,
    ), dim=-1)
    amplitude = 0.8 + 0.002 * time + 0.00001 * time.pow(3)
    phase = 0.1 * time + 0.0002 * time.pow(3)
    orbital_frequency = 0.15 - 0.0001 * (time - 30.0).square()
    radial_acceleration = 0.0008 + 0.00002 * time
    return time, state, amplitude, phase, orbital_frequency, radial_acceleration, step


def _result(targets=None):
    values = _synthetic()
    time, state, amplitude, phase, frequency, acceleration, step = values
    if targets is None:
        targets = published_2017_nqc_targets(
            torch.tensor(0.24, dtype=time.dtype), torch.tensor(0.0, dtype=time.dtype)
        )
    return assemble_teob_merger_nqc(
        time, state, amplitude, phase, frequency, acceleration,
        nu=torch.tensor(0.24, dtype=time.dtype), fixed_step=step,
        targets=targets, target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
    )


def test_quartic_derivative_reproduces_polynomial_at_all_boundaries():
    x = torch.arange(9, dtype=torch.float64) * 0.25 - 1.0
    polynomial = x.pow(4) - 2.0 * x.pow(3) + 0.5 * x.square() + 3.0 * x - 1.0
    expected = 4.0 * x.pow(3) - 6.0 * x.square() + x + 3.0
    derivative = quartic_derivative(polynomial, x[1] - x[0])
    batched = quartic_derivative(torch.stack((polynomial, 2.0 * polynomial)), x[1] - x[0])

    torch.testing.assert_close(derivative, expected, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(batched[0], expected, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(batched[1], 2.0 * expected, rtol=2e-13, atol=2e-12)


def test_quartic_derivative_preserves_an_explicit_sample_axis():
    x = torch.arange(7, dtype=torch.float64)
    polynomial = x.pow(4) - 2.0 * x.pow(3) + x.square() + 2.0 * x + 1.0
    expected = 4.0 * x.pow(3) - 6.0 * x.square() + 2.0 * x + 2.0
    values = torch.stack((polynomial, polynomial + 3.0), dim=0).unsqueeze(-1)
    result = quartic_derivative(values, 1.0, sample_dim=1)
    assert result.shape == values.shape
    torch.testing.assert_close(result[:, :, 0], expected.expand(2, -1))


def test_exact_and_off_knot_matching_indices_follow_searchsorted_rule():
    time = torch.arange(8, dtype=torch.float64) * 0.5
    assert matching_index(time, time[4]) == 3
    assert matching_index(time, time[4] - 0.1) == 2
    assert matching_index(time, time[4] + 0.1) == 3
    with pytest.raises(TEOBMergerNQCError, match="outside"):
        matching_index(time, -0.01)


def test_published_2017_provider_uses_printed_equal_mass_constants_and_domain():
    spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    equal = published_2017_nqc_targets(torch.tensor(0.25, dtype=torch.float64), spin)
    torch.testing.assert_close(
        torch.stack(equal),
        torch.tensor([0.31973334, -0.00195014, 0.38321834, 0.01190548], dtype=torch.float64),
        rtol=0.0, atol=5e-9,
    )
    unequal = published_2017_nqc_targets(torch.tensor(0.24, dtype=torch.float64), spin)
    assert not torch.equal(equal.amplitude, unequal.amplitude)
    objective = sum(value.square() for value in unequal)
    (gradient,) = torch.autograd.grad(objective, spin)
    assert torch.isfinite(gradient)
    with pytest.raises(ValueError, match="0.16 < nu"):
        published_2017_nqc_targets(torch.tensor(0.16, dtype=torch.float64), spin.detach())
    with pytest.raises(ValueError, match="0.16 < nu"):
        published_2017_nqc_targets(torch.tensor(0.26, dtype=torch.float64), spin.detach())


def test_published_2017_provider_matches_equal_mass_nonzero_spin_vector():
    """Check Table II away from the zero-spin point using printed decimals."""

    spin = torch.tensor(0.37, dtype=torch.float64)
    result = published_2017_nqc_targets(torch.tensor(0.25, dtype=torch.float64), spin)
    # Independently evaluated from the four Table-II polynomials in the
    # published 8-decimal form; this does not call the provider to construct
    # the expected values.
    expected = torch.tensor(
        [0.3201771704491095, -0.0014871050157753,
         0.4134932732180102, 0.0120137526330526],
        dtype=torch.float64,
    )
    torch.testing.assert_close(torch.stack(result), expected, rtol=0.0, atol=2e-15)


def test_published_2017_provider_matches_unequal_nonzero_spin_vector():
    """Check Table IV and Eq. (A3) away from the zero-spin point."""

    nu = torch.tensor(0.24, dtype=torch.float64)
    spin = torch.tensor(-0.41, dtype=torch.float64)
    result = published_2017_nqc_targets(nu, spin)
    # Independently evaluated from the printed Table-IV coefficients and
    # Eq. (A3), retaining their published precision.
    expected = torch.tensor(
        [0.3184597766437016, -0.0023814793150504,
         0.35048713380380108, 0.0111292305],
        dtype=torch.float64,
    )
    torch.testing.assert_close(torch.stack(result), expected, rtol=0.0, atol=2e-15)


def test_mode_phase_lifts_residual_before_unwrapping_sparse_carrier():
    time = torch.arange(10, dtype=torch.float64)
    orbital_phase = 2.0 * time
    phase = 2.0 * orbital_phase + 0.9 * time
    mode = (1.0 + 0.1 * time) * torch.exp(-1j * phase)
    recovered = phase_from_mode(mode, orbital_phase)
    torch.testing.assert_close(recovered, phase, rtol=0.0, atol=2e-12)


def test_assembly_uses_cut_exclusion_auxiliary_spacing_and_provenance():
    result = _result()
    assert result.cut_start_index == 40
    assert result.cut_last_index == 79
    assert result.auxiliary_spacing == torch.tensor(0.5, dtype=torch.float64)
    assert result.auxiliary.time[0] == result.time[result.cut_start_index]
    assert result.auxiliary.time[-1] <= result.time[result.cut_last_index]
    assert result.auxiliary.time[-1] < result.time[-1]
    assert result.target_provenance == PUBLISHED_2017_NQC_PROVENANCE
    assert result.auxiliary_negative_amplitude.shape == result.auxiliary.time.shape
    assert result.aux_amplitude_rw is result.auxiliary_amplitude_rw
    with pytest.raises(ValueError, match="explicit NQC targets"):
        time, state, amplitude, phase, frequency, acceleration, step = _synthetic()
        assemble_teob_merger_nqc(
            time, state, amplitude, phase, frequency, acceleration,
            nu=0.24, fixed_step=step, target_provenance="caller target",
        )
    with pytest.raises(ValueError, match="descriptive"):
        time, state, amplitude, phase, frequency, acceleration, step = _synthetic()
        assemble_teob_merger_nqc(
            time, state, amplitude, phase, frequency, acceleration,
            nu=0.24, fixed_step=step, targets=(0.3, 0.0, 0.4, 0.01),
            target_provenance=" ",
        )


def test_assembly_osculates_discrete_conditions_and_differentiates_product():
    result = _result()
    index = result.matching_index
    spacing = result.auxiliary_spacing
    auxiliary = result.auxiliary
    normalized_corrected = result.auxiliary_amplitude_rw * (
        1.0 + (auxiliary.basis_amplitude * result.coefficients.amplitude).sum(-1)
    )
    corrected_phase = auxiliary.phase - (
        auxiliary.basis_phase * result.coefficients.phase
    ).sum(-1)
    derivative = quartic_derivative(normalized_corrected, spacing)
    frequency = quartic_derivative(corrected_phase, spacing)
    frequency_rate = quartic_derivative(frequency, spacing)
    torch.testing.assert_close(
        normalized_corrected[index], result.targets.amplitude,
        rtol=2e-11, atol=2e-12,
    )
    torch.testing.assert_close(
        derivative[index], result.targets.amplitude_derivative,
        rtol=2e-11, atol=2e-12,
    )
    torch.testing.assert_close(frequency[index], result.targets.frequency,
                               rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(frequency_rate[index], result.targets.frequency_derivative,
                               rtol=2e-11, atol=2e-12)
    explicit_product_derivative = quartic_derivative(
        result.auxiliary_amplitude_rw.unsqueeze(-1) * auxiliary.basis_amplitude,
        spacing, sample_dim=-2,
    )[index]
    torch.testing.assert_close(
        explicit_product_derivative,
        result.match.amplitude_matrix[1],
        rtol=0.0, atol=2e-12,
    )
    # Use a deliberately non-polynomial product to make the discrete-stencil
    # distinction visible independently of the physical basis values.
    grid = torch.arange(11, dtype=torch.float64) * spacing
    left = torch.sin(0.8 * grid)
    right = torch.cos(grid)
    discrete_product = quartic_derivative(left * right, spacing)
    continuous_product_rule = (
        quartic_derivative(left, spacing) * right
        + left * quartic_derivative(right, spacing)
    )
    assert not torch.allclose(discrete_product[2:-2], continuous_product_rule[2:-2],
                              rtol=1e-5, atol=1e-10)
    torch.testing.assert_close(result.match.amplitude_residual,
                               torch.zeros(2, dtype=torch.float64), atol=3e-11, rtol=0)
    torch.testing.assert_close(result.match.phase_residual,
                               torch.zeros(2, dtype=torch.float64), atol=3e-11, rtol=0)


def test_signed_auxiliary_amplitude_is_exposed_without_absolute_value():
    targets = TEOBMergerNQCTargets(0.1, 0.01, 0.1, 0.001)
    result = _result(targets)
    assert torch.equal(
        result.auxiliary_negative_amplitude,
        result.auxiliary.corrected_amplitude < 0,
    )
    assert bool(result.auxiliary_negative_amplitude.any())
    assert bool((result.auxiliary.corrected_amplitude < 0).any())
    torch.testing.assert_close(result.auxiliary.mode.abs(),
                               result.auxiliary.corrected_amplitude.abs())


def test_singular_matching_basis_fails_without_regularization():
    time, state, amplitude, phase, frequency, acceleration, step = _synthetic()
    state = state.clone()
    state[:, 2] = 0.0
    acceleration = torch.zeros_like(acceleration)
    targets = TEOBMergerNQCTargets(0.3, 0.0, 0.4, 0.0)
    with pytest.raises(TEOBMergerNQCError, match="singular|ill-conditioned"):
        assemble_teob_merger_nqc(
            time, state, amplitude, phase, frequency, acceleration,
            nu=0.24, fixed_step=step, targets=targets,
            target_provenance="independent synthetic target",
        )


def test_assembly_preserves_finite_gradients_through_sampled_data_and_targets():
    time, state, amplitude, phase, frequency, acceleration, step = _synthetic()
    amplitude = amplitude.clone().requires_grad_()
    phase = phase.clone().requires_grad_()
    target_amplitude = torch.tensor(0.32, dtype=time.dtype, requires_grad=True)
    targets = TEOBMergerNQCTargets(target_amplitude, torch.tensor(-0.002, dtype=time.dtype),
                                   torch.tensor(0.38, dtype=time.dtype), torch.tensor(0.01, dtype=time.dtype))
    result = assemble_teob_merger_nqc(
        time, state, amplitude, phase, frequency, acceleration,
        nu=0.24, fixed_step=step, targets=targets, target_provenance="gradient target",
    )
    objective = result.coefficients.amplitude.square().sum() + result.coefficients.phase.square().sum()
    gradients = torch.autograd.grad(objective, (amplitude, phase, target_amplitude), allow_unused=False)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_non_scalar_intrinsic_and_invalid_domains_are_rejected():
    values = _synthetic()
    time, state, amplitude, phase, frequency, acceleration, step = values
    with pytest.raises(ValueError, match="scalar-intrinsic"):
        assemble_teob_merger_nqc(
            time.unsqueeze(0), state, amplitude, phase, frequency, acceleration,
            nu=0.24, fixed_step=step, targets=(0.3, 0.0, 0.4, 0.01),
            target_provenance="test target",
        )
    with pytest.raises(TEOBMergerNQCError, match="positive radius"):
        bad_state = state.clone()
        bad_state[-1, 0] = 0.0
        assemble_teob_merger_nqc(
            time, bad_state, amplitude, phase, frequency, acceleration,
            nu=0.24, fixed_step=step, targets=(0.3, 0.0, 0.4, 0.01),
            target_provenance="test target",
        )
