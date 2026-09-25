"""Cross-backend checks for the development-only full-series v4 NQC rule."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.eob.jax.numerical_attachment_clean import (
    calibrate_nqc_22_full_series as jax_calibrate,
)
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_targets as jax_targets
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline as jax_spline
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets as torch_targets
from torchwave.approximants.eob.numerical_attachment import (
    calibrate_nqc_22_full_series as torch_calibrate,
    natural_cubic_spline as torch_spline,
)

jax.config.update("jax_enable_x64", True)


def test_scanned_natural_spline_matches_torch_on_unequal_knots_and_jit():
    knots = np.array([0, 0.13, 0.8, 1.9, 2.05, 3.7], dtype=np.float64)
    values = np.stack((np.sin(knots) + knots**2, np.exp(-knots) + 0.2 * knots), axis=-1)
    query = np.array([0.07, 0.4, 1.2, 1.97, 3.2], dtype=np.float64)
    expected = torch_spline(torch.tensor(knots), torch.tensor(values), scalar_output=False)
    for derivative in (0, 1, 2):
        actual = jax.jit(lambda x, y: jax_spline(x, y, scalar_output=False).evaluate(
            jnp.asarray(query), derivative))(jnp.asarray(knots), jnp.asarray(values))
        np.testing.assert_allclose(np.asarray(actual),
                                   np.asarray(expected.evaluate(torch.tensor(query), derivative)),
                                   rtol=2e-13, atol=2e-13)


def test_full_series_nqc_matches_torch_for_nontrivial_wrapped_mode():
    time = np.linspace(0, 30, 121)
    delta = time - 18
    nu, chi = 0.22, 0.1
    target = torch_targets(torch.tensor(nu, dtype=torch.float64),
                           torch.tensor(chi, dtype=torch.float64))
    amplitude = (float(target.amplitude) + 1e-4 * delta**2 + 5e-6 * delta**4)
    frequency = float(target.frequency) + float(target.frequency_derivative) * delta + 1e-6 * delta**2
    phase = (2.7 + float(target.frequency) * delta
             + 0.5 * float(target.frequency_derivative) * delta**2
             + 1e-6 / 3 * delta**3 + 1e-3 * delta + 1e-5 * delta**2)
    mode = amplitude * np.exp(-1j * phase)
    r = 8 - 0.025 * time + 0.00015 * time**2
    pr = -0.02 - 0.0005 * time + 1e-5 * time**2 + 0.001 * np.sin(0.37 * time)
    omega = 0.08 + 0.002 * time + 0.0002 * time**2 - 4e-6 * time**3

    def t(value):
        return torch.as_tensor(value, dtype=torch.float64)

    expected = torch_calibrate(t(time), t(r), t(pr), t(omega), t(amplitude),
                               t(frequency), t(18.0), target,
                               base_mode=torch.as_tensor(mode, dtype=torch.complex128))
    actual = jax_calibrate(jnp.asarray(time), jnp.asarray(r), jnp.asarray(pr),
                           jnp.asarray(omega), jnp.asarray(amplitude),
                           jnp.asarray(frequency), jnp.asarray(18.0),
                           jax_targets(nu, chi), base_mode=jnp.asarray(mode))

    def close(left, right, *, rtol=2e-9, atol=2e-11):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=rtol, atol=atol)

    close(actual.sampled_waveform_phase - actual.sampled_waveform_phase[0],
          expected.sampled_waveform_phase - expected.sampled_waveform_phase[0])
    close(actual.amplitude_matching_matrix, expected.amplitude_matching_matrix)
    close(actual.phase_matching_matrix, expected.phase_matching_matrix)
    close(actual.coefficients.amplitude, expected.coefficients.amplitude)
    close(actual.coefficients.phase, expected.coefficients.phase)
    close(actual.calibration.amplitude_condition, expected.calibration.amplitude_condition)
    close(actual.calibration.phase_condition, expected.calibration.phase_condition)
    close(actual.sampled_corrected_amplitude, expected.sampled_corrected_amplitude)
    close(actual.sampled_corrected_phase - actual.sampled_corrected_phase[0],
          expected.sampled_corrected_phase - expected.sampled_corrected_phase[0])
    query = jnp.array([17.31, 18.0, 18.37])
    close(actual.evaluate(query), expected.evaluate(t(np.array(query))))
    close(actual.evaluate(jnp.asarray(18.0)), expected.evaluate(t(18.0)))
