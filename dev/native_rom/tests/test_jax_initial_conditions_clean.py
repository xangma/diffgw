"""Development-only JAX quasicircular initial-state value and gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchSourceCalibratedHistoricalHamiltonian,
)
from jaxwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_initial_conditions,
)
from torchwave.approximants.eob.seobnrv4_flux import (
    SEOBNRv4PublishedFlux as TorchSEOBNRv4PublishedFlux,
)

jax.config.update("jax_enable_x64", True)


def test_q4_initial_state_and_spin_derivative_match_torch_and_finite_difference():
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

    torch_spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    torch_hamiltonian = TorchSourceCalibratedHistoricalHamiltonian(
        torch.tensor(0.16, dtype=torch.float64), torch_spin,
        torch.tensor(0.0, dtype=torch.float64),
        delta=torch.tensor(0.6, dtype=torch.float64),
    )
    torch_flux = TorchSEOBNRv4PublishedFlux(**SOURCE_FLUX_OPTIONS)
    torch_value = torch.stack(torch_initial_conditions(
        torch_hamiltonian, torch_flux, torch.tensor(0.04, dtype=torch.float64)
    ))
    torch_derivative = torch.stack([
        torch.autograd.grad(value, torch_spin, retain_graph=True)[0]
        for value in torch_value
    ])
    np.testing.assert_allclose(compiled(0.0), torch_value.detach(), rtol=0, atol=3e-8)
    np.testing.assert_allclose(derivative, torch_derivative.detach(), rtol=0, atol=1e-8)
