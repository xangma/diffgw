"""The long-waveform route must preserve the frozen local matching diagnostic."""

import torch

from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    _polynomial_amplitude_derivatives,
    _polynomial_phase_matrix,
    _polynomial_rhs_derivatives,
    _stationary_orbital_peak_time,
)
from dev.native_rom.eob.torch.local_nqc_control import (
    local_nqc_controls,
    stationary_peak_time,
)
from torchwave.approximants.eob.numerical_attachment import (
    _eval_batch_scalar,
    natural_cubic_spline,
)


def test_combined_fit_matches_frozen_columns_and_stationary_peak():
    time = torch.arange(0, 10.001, 0.01, dtype=torch.float64)
    peak = 5.137
    omega = 0.11 - 0.0008 * (time - peak)**2
    state = torch.stack((9 - 0.01 * time, 0.12 * time,
                         -0.001 - 0.0002 * time, 3.1 + 0.002 * time), dim=-1)
    amplitude = 0.2 + 0.003 * time + 0.0001 * time**2
    phase = 0.2 * time + 0.0008 * time**2
    mode = torch.polar(amplitude, -phase)
    spline = natural_cubic_spline(time, omega)
    reference_peak = _stationary_orbital_peak_time(
        time.numpy(), omega.numpy(),
        lambda t: _eval_batch_scalar(spline, time.new_tensor(t), derivative=1),
        lambda t: _eval_batch_scalar(spline, time.new_tensor(t), derivative=2))
    assert abs(float(stationary_peak_time(time, omega)) - reference_peak) < 1e-12

    match = time.new_tensor(peak + 0.13)
    options = {"half_width_M": 2.0, "degree": 4}
    actual = local_nqc_controls(time, state, omega, mode, match, **options)
    reference = (
        _polynomial_amplitude_derivatives(time, state, omega, mode, match, **options),
        _polynomial_phase_matrix(time, state, omega, match, **options),
        _polynomial_rhs_derivatives(time, mode, match, **options),
    )
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
