"""Selected q4 Torch/JAX pathwise base-mode derivative parity."""

import numpy as np
import torch

from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.adaptive_driver_clean import fixed_fehlberg45_step
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


def test_q4_torch_pathwise_gradient_matches_frozen_jax_result():
    """Hold the 21 accepted times fixed; exclude controller/event derivatives."""
    chi1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    ham = SourceCalibratedHistoricalHamiltonian(
        torch.tensor(0.16, dtype=torch.float64), chi1,
        torch.tensor(0.0, dtype=torch.float64),
        delta=torch.tensor(0.6, dtype=torch.float64),
    )
    flux = SEOBNRv4PublishedFlux(**SOURCE_FLUX_OPTIONS)
    radius, radial, angular = find_quasicircular_initial_conditions(
        ham, flux, torch.tensor(0.04, dtype=torch.float64)
    )
    state = torch.stack((radius, torch.zeros_like(radius), radial, angular))
    field = EOBTrajectoryIntegrator(ham, flux).vector_field
    accepted_time = np.r_[0.0, 0.1, np.linspace(0.3, 3.9, 19), 4.0]
    assert accepted_time.size - 1 == 21
    for start, end in zip(accepted_time[:-1], accepted_time[1:]):
        trial = fixed_fehlberg45_step(
            field, torch.tensor(start, dtype=torch.float64), state,
            torch.tensor(end - start, dtype=torch.float64),
            atol=1e-13, rtol=1e-12,
        )
        assert trial.accepted
        state = trial.fifth
    mode, frequency = published_mode22_and_frequency(
        state, ham, flux, **(MODE_OPTIONS | {"retain_pan_cubic_spin": True})
    )
    value = torch.stack((mode.real, mode.imag, frequency))
    derivative = torch.stack([
        torch.autograd.grad(component, chi1, retain_graph=True)[0]
        for component in value
    ])
    # Independently compiled JAX values from the frozen 2026-09-25 plan.
    jax_value = [-0.08939993894712825, 0.0653614599894306, 0.08037388323460629]
    jax_derivative = [
        0.005481510690164895, -0.003948012910054575, -0.00023974626233679543,
    ]
    np.testing.assert_allclose(value.detach(), jax_value, rtol=0, atol=1e-12)
    np.testing.assert_allclose(derivative.detach(), jax_derivative, rtol=0, atol=1e-12)
