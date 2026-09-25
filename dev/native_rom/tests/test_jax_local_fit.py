"""Development-only value and differentiation checks for the local NQC fit."""

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    _jax_polynomial_nqc_controls,
    _jax_polynomial_nqc_controls_numpy,
)

jax.config.update("jax_enable_x64", True)


def test_local_polynomial_fit_is_jittable_and_differentiable():
    time = jnp.linspace(-3.0, 3.0, 97)
    state = jnp.column_stack((10.0 - 0.1*time, jnp.zeros_like(time),
                              -0.01 - 0.0002*time, 3.5 + 0.002*time))
    frequency = 0.03 + 0.0001*time
    phase = 0.7 + 0.06*time + 0.001*time**2

    def fit(amplitude_offset):
        amplitude = 0.2 + amplitude_offset + 0.004*time + 0.0002*time**2
        mode = amplitude*jnp.exp(-1j*phase)
        return _jax_polynomial_nqc_controls(
            time, state, frequency, mode, 0.1, half_width_M=2.0, degree=4)

    eager = fit(0.0)
    compiled = jax.jit(fit)(0.0)
    reference_mode = (0.2 + 0.004*time + 0.0002*time**2)*jnp.exp(-1j*phase)
    reference = _jax_polynomial_nqc_controls_numpy(
        time, state, frequency, reference_mode, 0.1,
        half_width_M=2.0, degree=4)
    for actual, compiled_actual, expected in zip(eager, compiled, reference):
        np.testing.assert_allclose(actual, compiled_actual, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    tangent = jax.jacfwd(fit)(0.0)
    for array in tangent:
        assert np.isfinite(np.asarray(array)).all()
