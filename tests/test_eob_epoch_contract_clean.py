"""Small numerical checks for the clean epoch-coordinate diagnostics."""

import numpy as np

from scripts.diagnose_eob_epoch_contract_clean import (
    _postmatch_residual,
    local_polynomial_peak,
    quadratic_peak,
)


def test_peak_diagnostics_preserve_epoch_translation_separately():
    relative = np.arange(9, dtype=np.float64)
    epoch = -5002.7
    ordinates = 2.0 - (relative - 4.25) ** 2
    relative_peak, relative_offset = quadratic_peak(relative, ordinates, 4)
    physical_peak, physical_offset = quadratic_peak(
        epoch + relative, ordinates, 4
    )
    assert np.isclose(physical_peak - relative_peak, epoch)
    assert np.isclose(physical_offset, relative_offset)


def test_degree_six_stencil_is_a_diagnostic_on_a_known_polynomial():
    time = np.arange(-3.0, 4.0)
    center = 0.2
    x = time - center
    values = 2.0 - 0.3 * x**2 + 0.05 * x**4 - 0.002 * x**6
    result = local_polynomial_peak(time, values, 3, 6)
    assert result["resolved"]
    assert np.isclose(result["time"], center, atol=1e-11)


def test_postmatch_residual_reports_epoch_support_without_shifts():
    output_time = np.arange(6, dtype=np.float64)
    public = np.exp(1j * output_time)
    residual = _postmatch_residual(
        public.copy(), public, output_time, 2.0, epoch_M=-10.0
    )
    assert residual["available"]
    assert residual["time_shift_applied_M"] == 0.0
    assert residual["phase_shift_applied_rad"] == 0.0
    assert residual["support_M_initial_origin"] == [2.0, 5.0]
    assert residual["public_support_M_epoch_coordinate"] == [-8.0, -5.0]
    assert residual["raw_complex_l2"] == 0.0
