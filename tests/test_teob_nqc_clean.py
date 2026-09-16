import pytest
import torch

from torchwave.approximants.eob.teob_nqc_clean import (
    NQCBasis,
    NQCBasisValues,
    NQCCoefficients,
    NQCMatchingTargets,
    NQCSignal,
    apply_nqc_correction,
    corrected_matching_signal,
    make_nqc_basis,
    match_nqc,
    merger_chart_time_from_nqc,
    nqc_basis_values,
    nqc_correction_factor,
    nqc_match_time_from_orbital_peak,
    rw_denormalize_22,
    rw_normalize_22,
    solve_nqc_coefficients,
)


def _synthetic(dtype=torch.float64):
    basis = NQCBasis(
        amplitude=torch.tensor([[0.20, -0.10], [0.25, 0.11]], dtype=dtype),
        amplitude_derivative=torch.tensor([[0.04, 0.07], [-0.03, 0.09]], dtype=dtype),
        phase=torch.tensor([[0.15, -0.22], [0.11, 0.19]], dtype=dtype),
        phase_derivative=torch.tensor([[0.70, -0.40], [0.20, 0.50]], dtype=dtype),
        phase_second_derivative=torch.tensor([[0.03, 0.08], [-0.06, 0.12]], dtype=dtype),
    )
    base = NQCSignal(
        torch.tensor([1.40, 0.90], dtype=dtype),
        torch.tensor([0.10, -0.06], dtype=dtype),
        torch.tensor([0.40, 0.52], dtype=dtype),
        torch.tensor([0.02, -0.03], dtype=dtype),
    )
    true = NQCCoefficients(
        torch.tensor([0.35, -0.12], dtype=dtype),
        torch.tensor([-0.20, 0.27], dtype=dtype),
    )
    target = NQCMatchingTargets(*corrected_matching_signal(base, basis, true))
    return base, basis, true, target


def test_basis_values_are_eq97_and_derivatives_are_explicit():
    dtype = torch.float64
    values = nqc_basis_values(
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(-0.3, dtype=dtype),
        torch.tensor(0.4, dtype=dtype),
        torch.tensor(0.5, dtype=dtype),
    )
    torch.testing.assert_close(
        values.amplitude, torch.tensor([0.140625, 1.5625], dtype=dtype)
    )
    torch.testing.assert_close(
        values.phase, torch.tensor([-0.375, -0.24], dtype=dtype)
    )
    assert values.n is values.amplitude
    assert values.n_prime is values.phase

    basis = make_nqc_basis(
        values,
        torch.tensor([0.1, 0.2], dtype=dtype),
        torch.tensor([0.3, -0.4], dtype=dtype),
        torch.tensor([0.5, 0.6], dtype=dtype),
    )
    assert basis.amplitude_derivative.dtype == dtype
    torch.testing.assert_close(basis.dn_dt, basis.amplitude_derivative)
    torch.testing.assert_close(basis.dn_prime_dt, basis.phase_derivative)
    torch.testing.assert_close(basis.d2n_prime_dt2, basis.phase_second_derivative)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_four_conditions_recover_synthetic_coefficients(dtype):
    base, basis, true, targets = _synthetic(dtype)
    result = match_nqc(base, targets, basis)
    expected_amplitude = true.amplitude.expand_as(result.coefficients.amplitude)
    expected_phase = true.phase.expand_as(result.coefficients.phase)
    torch.testing.assert_close(result.coefficients.amplitude, expected_amplitude,
                               rtol=2e-5 if dtype == torch.float32 else 1e-11,
                               atol=2e-6 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(result.coefficients.phase, expected_phase,
                               rtol=2e-5 if dtype == torch.float32 else 1e-11,
                               atol=2e-6 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(result.amplitude_residual, torch.zeros_like(result.amplitude_residual),
                               atol=2e-6 if dtype == torch.float32 else 1e-12,
                               rtol=0)
    torch.testing.assert_close(result.phase_residual, torch.zeros_like(result.phase_residual),
                               atol=2e-6 if dtype == torch.float32 else 1e-12,
                               rtol=0)
    recovered = corrected_matching_signal(base, basis, result.coefficients)
    for actual, expected in zip(recovered, targets):
        torch.testing.assert_close(actual, expected,
                                   rtol=2e-5 if dtype == torch.float32 else 1e-11,
                                   atol=2e-6 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(solve_nqc_coefficients(base, targets, basis).amplitude,
                               result.coefficients.amplitude)


def test_correction_uses_signed_amplitude_and_plus_i_phase():
    dtype = torch.float64
    basis = NQCBasisValues(
        torch.tensor([[0.20, 0.30], [0.40, -0.10]], dtype=dtype),
        torch.tensor([[0.50, -0.20], [-0.30, 0.70]], dtype=dtype),
    )
    coefficients = NQCCoefficients(
        torch.tensor([[-6.0, 0.0], [0.25, -0.5]], dtype=dtype),
        torch.tensor([[0.4, 0.2], [-0.1, 0.3]], dtype=dtype),
    )
    waveform = torch.ones(2, dtype=torch.complex128)
    expected = torch.stack((
        (-0.2) * torch.exp(1j * torch.tensor(0.16, dtype=dtype)),
        (1.0 + 0.25 * 0.4 + (-0.5) * (-0.1))
        * torch.exp(1j * torch.tensor(0.24, dtype=dtype)),
    ))
    corrected = apply_nqc_correction(waveform, basis, coefficients)
    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(nqc_correction_factor(basis, coefficients), expected)


def test_correction_sample_axes_are_explicit():
    dtype = torch.float64
    n = torch.tensor([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
                      [[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]], dtype=dtype)
    p = torch.tensor([[[0.2, -0.1], [0.3, -0.2], [0.4, -0.3]],
                      [[-0.1, 0.2], [-0.2, 0.3], [-0.3, 0.4]]], dtype=dtype)
    basis = NQCBasisValues(n, p)
    coefficients = NQCCoefficients(
        torch.tensor([[0.2, -0.1], [0.1, 0.3]], dtype=dtype),
        torch.tensor([[0.4, 0.2], [-0.3, 0.1]], dtype=dtype),
    )
    waveform = torch.ones(2, 3, dtype=torch.complex128)
    factor = nqc_correction_factor(basis, coefficients, sample_dims=1)
    assert factor.shape == waveform.shape
    torch.testing.assert_close(apply_nqc_correction(
        waveform, basis, coefficients, sample_dims=1), waveform * factor)


def test_common_amplitude_rescaling_does_not_change_coefficients():
    base, basis, true, targets = _synthetic()
    scale = 7.0
    scaled_base = NQCSignal(base.amplitude * scale, base.amplitude_derivative * scale,
                            base.frequency, base.frequency_derivative)
    scaled_targets = NQCMatchingTargets(
        targets.amplitude * scale, targets.amplitude_derivative * scale,
        targets.frequency, targets.frequency_derivative)
    result = solve_nqc_coefficients(scaled_base, scaled_targets, basis)
    torch.testing.assert_close(result.amplitude, true.amplitude.expand_as(result.amplitude))
    torch.testing.assert_close(result.phase, true.phase.expand_as(result.phase))


def test_rw_normalization_is_exactly_sqrt_24_and_round_trips():
    mode = torch.tensor([2.0 + 3.0j, -1.0 + 0.5j], dtype=torch.complex128)
    normalized = rw_normalize_22(mode)
    torch.testing.assert_close(rw_denormalize_22(normalized), mode)
    torch.testing.assert_close(
        rw_normalize_22(torch.tensor([torch.sqrt(torch.tensor(24.0)) + 0j])),
        torch.ones(1, dtype=torch.complex64),
    )


def test_degenerate_systems_and_invalid_basis_fail_loudly():
    base, basis, _, targets = _synthetic()
    singular_amplitude = basis._replace(
        amplitude=torch.tensor([[0.2, 0.2], [0.25, 0.25]], dtype=torch.float64),
        amplitude_derivative=torch.tensor([[0.04, 0.04], [-0.03, -0.03]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="rank deficient|ill-conditioned"):
        match_nqc(base, targets, singular_amplitude)

    singular_phase = basis._replace(
        phase_derivative=torch.tensor([[0.7, 0.7], [0.2, 0.2]], dtype=torch.float64),
        phase_second_derivative=torch.tensor([[0.03, 0.03], [-0.06, -0.06]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="rank deficient|ill-conditioned"):
        match_nqc(base, targets, singular_phase)

    with pytest.raises(ValueError, match="r > 0"):
        nqc_basis_values(*(torch.tensor(x, dtype=torch.float64) for x in (0.0, 0.1, 0.4, 0.2)))
    with pytest.raises(ValueError, match="nonzero orbital frequency"):
        nqc_basis_values(*(torch.tensor(x, dtype=torch.float64) for x in (2.0, 0.1, 0.0, 0.2)))


def test_dtype_device_and_autograd_are_preserved():
    dtype = torch.float64
    r = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    pr = torch.tensor(-0.2, dtype=dtype, requires_grad=True)
    omega = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    acceleration = torch.tensor(0.4, dtype=dtype, requires_grad=True)
    values = nqc_basis_values(r, pr, omega, acceleration)
    assert values.amplitude.dtype == dtype
    assert values.amplitude.device == r.device
    basis = NQCBasis(
        values.amplitude,
        torch.tensor([0.05, 0.09], dtype=dtype),
        values.phase,
        torch.tensor([0.2, 0.4], dtype=dtype),
        torch.tensor([0.1, -0.3], dtype=dtype),
    )
    base = NQCSignal(
        torch.tensor(1.2, dtype=dtype, requires_grad=True),
        torch.tensor(0.1, dtype=dtype, requires_grad=True),
        torch.tensor(0.5, dtype=dtype, requires_grad=True),
        torch.tensor(0.02, dtype=dtype, requires_grad=True),
    )
    targets = NQCMatchingTargets(
        torch.tensor(1.3, dtype=dtype), torch.tensor(0.08, dtype=dtype),
        torch.tensor(0.48, dtype=dtype), torch.tensor(0.01, dtype=dtype),
    )
    result = match_nqc(base, targets, basis)
    loss = result.coefficients.amplitude.square().sum() + result.coefficients.phase.square().sum()
    gradients = torch.autograd.grad(loss, (r, pr, omega, acceleration, *base), allow_unused=False)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_time_conventions_preserve_dtype_device_and_gradient():
    peak = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    nqc_time = nqc_match_time_from_orbital_peak(peak)
    merger = merger_chart_time_from_nqc(nqc_time)
    torch.testing.assert_close(nqc_time, torch.tensor([2.0, 3.0], dtype=peak.dtype))
    torch.testing.assert_close(merger, torch.tensor([0.0, 1.0], dtype=peak.dtype))
    merger.sum().backward()
    torch.testing.assert_close(peak.grad, torch.ones_like(peak))


def test_wrong_types_are_rejected():
    with pytest.raises(TypeError, match="torch tensors"):
        nqc_basis_values(2.0, torch.tensor(0.1), torch.tensor(0.4), torch.tensor(0.2))
    with pytest.raises(TypeError, match="complex64 or complex128"):
        apply_nqc_correction(torch.ones(1), NQCBasisValues(
            torch.ones(1, 2), torch.ones(1, 2)), NQCCoefficients(
                torch.zeros(1, 2), torch.zeros(1, 2)))
