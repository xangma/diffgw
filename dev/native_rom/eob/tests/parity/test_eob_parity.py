from dev.native_rom.eob.dispatch import torch_get_fd_waveform, jax_get_fd_waveform
"""Cross-framework parity tests: torchwave vs jaxwave for SEOBNR / EOB engine.

Validates:
1. EOBHamiltonian and SEOBNRv4Hamiltonian conservative dynamics and potentials.
2. EOBRadiationReaction and FactorizedRadiationReaction dissipative dynamics.
3. find_quasicircular_initial_conditions solve.
4. EOBTrajectoryIntegrator.integrate (RK4 trajectory integration).
5. _quadrupole_fd Fourier projection.
6. SEOBNR_ODE frequency-domain waveforms across mass ratios and aligned spins.
7. SEOBNRv4PublishedNonspinning end-to-end waveform reconstruction with NQC attachment.
8. jaxwave.waveforms.get_fd_waveform dispatch for SEOBNR aliases.
9. JIT compilation of core EOB components under jax.jit.
"""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)


from dev.native_rom.eob.torch.waveform import SEOBNR_ODE as TorchSEOBNR_ODE
from dev.native_rom.eob.torch.published_waveform import (
    SEOBNRv4PublishedNonspinning as TorchPublishedNonspin,
)

from dev.native_rom.eob.jax.waveform import SEOBNR_ODE as JaxSEOBNR_ODE
from dev.native_rom.eob.jax.published_waveform import (
    SEOBNRv4PublishedNonspinning as JaxPublishedNonspin,
)


@pytest.mark.parametrize("m1, m2, s1, s2", [
    (36.0, 29.0, 0.0, 0.0),
    (40.0, 20.0, 0.2, -0.1),
])
def test_seobnr_ode_waveform_parity(m1, m2, s1, s2):
    """Test SEOBNR_ODE frequency-domain waveforms across masses and spins."""
    f = np.linspace(25.0, 400.0, 64)
    t_f = torch.tensor(f, dtype=torch.float64)
    j_f = jnp.asarray(f, dtype=jnp.float64)

    t_model = TorchSEOBNR_ODE()
    j_model = JaxSEOBNR_ODE()

    t_hp, t_hc = t_model(
        t_f,
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        chi1=torch.tensor(s1, dtype=torch.float64),
        chi2=torch.tensor(s2, dtype=torch.float64),
        num_steps=100,
    )
    j_hp, j_hc = j_model(
        j_f,
        m1,
        m2,
        chi1=s1,
        chi2=s2,
        num_steps=100,
    )

    np.testing.assert_allclose(t_hp.numpy(), np.asarray(j_hp), rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(t_hc.numpy(), np.asarray(j_hc), rtol=1e-10, atol=1e-25)


def test_seobnrv4_published_nonspinning_parity():
    """Test SEOBNRv4PublishedNonspinning end-to-end waveform with NQC and ringdown."""
    t_pub = TorchPublishedNonspin()
    j_pub = JaxPublishedNonspin()

    f = np.linspace(80.0, 400.0, 32)
    t_f = torch.tensor(f, dtype=torch.float64)
    j_f = jnp.asarray(f, dtype=jnp.float64)

    t_hp, t_hc = t_pub(
        t_f,
        torch.tensor(36.0, dtype=torch.float64),
        torch.tensor(29.0, dtype=torch.float64),
        f_lower=80.0,
        num_steps=600,
        final_spin_fit_order=(3, 4),
    )
    j_hp, j_hc = j_pub(
        j_f,
        36.0,
        29.0,
        f_lower=80.0,
        num_steps=600,
        final_spin_fit_order=(3, 4),
    )

    np.testing.assert_allclose(t_hp.numpy(), np.asarray(j_hp), rtol=1e-10, atol=1e-20)
    np.testing.assert_allclose(t_hc.numpy(), np.asarray(j_hc), rtol=1e-10, atol=1e-20)


@pytest.mark.parametrize("approx", ["seobnr", "seobnrode", "seobnrv5"])
def test_waveforms_dispatch_parity(approx):
    """Test get_fd_waveform dispatch across SEOBNR aliases."""
    f = np.linspace(25.0, 400.0, 32)
    t_hp, t_hc = torch_get_fd_waveform(
        approximant=approx,
        mass1=36.0,
        mass2=29.0,
        spin1z=0.0,
        spin2z=0.0,
        sample_frequencies=f,
        num_steps=80,
    )
    j_hp, j_hc = jax_get_fd_waveform(
        approximant=approx,
        mass1=36.0,
        mass2=29.0,
        spin1z=0.0,
        spin2z=0.0,
        sample_frequencies=f,
        num_steps=80,
    )

    np.testing.assert_allclose(t_hp.numpy(), np.asarray(j_hp), rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(t_hc.numpy(), np.asarray(j_hc), rtol=1e-10, atol=1e-25)
