"""Development-only JAX quasicircular initial-state value and gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from jaxwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux

jax.config.update("jax_enable_x64", True)


def test_q4_initial_state_compiles_and_spin_derivative_matches_finite_difference():
    flux = SEOBNRv4PublishedFlux(**SOURCE_FLUX_OPTIONS)

    def hamiltonian(chi1):
        return SourceCalibratedHistoricalHamiltonian(0.16, chi1, 0.0, delta=0.6)

    def initial(chi1):
        return jnp.stack(traceable_quasicircular_initial_conditions(
            hamiltonian(chi1), flux, 0.04, max_iter=8
        ))

    expected = np.asarray(find_quasicircular_initial_conditions(
        hamiltonian(jnp.asarray(0.0)), flux, jnp.asarray(0.04)
    ))
    compiled = jax.jit(initial)
    np.testing.assert_allclose(compiled(0.0), expected, rtol=0, atol=3e-8)
    np.testing.assert_allclose(compiled(0.0), initial(0.0), rtol=0, atol=1e-10)

    derivative = np.asarray(jax.jit(jax.jacfwd(initial))(0.0))
    assert np.isfinite(derivative).all()
    for step in (1e-4, 1e-5):
        finite_difference = (
            np.asarray(compiled(step)) - np.asarray(compiled(-step))
        ) / (2 * step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=0, atol=1e-7)
