"""Physical checks for the development-only native SEOBNRv4 warm start."""

import torch
import pytest

from dev.native_rom.eob.torch.warm_start import native_frequency_warm_start
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


@pytest.mark.parametrize("chi1, chi2", [(0.0, 0.0), (0.2, -0.1)])
def test_frequency_crossing_gradient_matches_centered_difference(chi1, chi2):
    torch.set_num_threads(1)
    dtype = torch.float64
    hamiltonian = SEOBNRv4Hamiltonian(
        torch.tensor(0.24, dtype=dtype), torch.tensor(chi1, dtype=dtype),
        torch.tensor(chi2, dtype=dtype), delta=torch.tensor(0.2, dtype=dtype))
    flux = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=True)
    start = torch.tensor(10 ** -1.5, dtype=dtype)
    target = torch.tensor(0.04, dtype=dtype, requires_grad=True)

    crossing = native_frequency_warm_start(hamiltonian, flux, start, target)
    assert abs(float(crossing.frequency_residual.detach())) <= 1e-11
    assert float(crossing.state[1].detach()) == 0.0
    radius_gradient = torch.autograd.grad(
        crossing.state[0], target, retain_graph=True)[0]
    time_gradient = torch.autograd.grad(crossing.crossing_time, target)[0]

    epsilon = 1e-6
    with torch.no_grad():
        lower = native_frequency_warm_start(
            hamiltonian, flux, start, target.detach()-epsilon)
        upper = native_frequency_warm_start(
            hamiltonian, flux, start, target.detach()+epsilon)
    radius_difference = (upper.state[0]-lower.state[0])/(2*epsilon)
    time_difference = (upper.crossing_time-lower.crossing_time)/(2*epsilon)
    torch.testing.assert_close(radius_gradient, radius_difference, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(time_gradient, time_difference, rtol=2e-4, atol=1e-3)
