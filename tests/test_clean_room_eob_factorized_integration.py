"""Independent energy-balance and covariance checks for factorized EOB wiring."""

import pytest
import torch

from torchwave.approximants.eob.hamiltonian import Nonspinning3PNHamiltonian
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.radiation_reaction import FactorizedRadiationReaction


DTYPE = torch.float64


@pytest.mark.parametrize("model,spin", [("din2009_nonspinning", 0.), ("pan2011_leading_spin", .4)])
def test_circular_mechanical_power_equals_emitted_multipolar_flux(model, spin):
    nu = torch.tensor(.2, dtype=DTYPE)
    ham = SEOBNRv4Hamiltonian(nu, nu*0+spin, nu*0-spin/2)
    rr = FactorizedRadiationReaction(model)
    # A radial turning point makes the radial dissipative work zero. The
    # remaining mechanical power is derived from H_real, not the torque API.
    state = torch.tensor([12., .3, 0., 4.], dtype=DTYPE, requires_grad=True)
    real = ham.real_hamiltonian_from_eff(ham.effective_hamiltonian(state[0], state[2], state[3]))
    gradient = torch.autograd.grad(real, state)[0]
    field = EOBTrajectoryIntegrator(ham, rr).vector_field(state)
    flux = rr.energy_flux(field[1], state[2], state[3], r=state[0], hamiltonian=ham)
    torch.testing.assert_close((gradient*field).sum(), -flux, rtol=3e-13, atol=1e-18)
    assert flux > 0 and field[3] < 0


def test_factorized_adiabatic_initial_data_recovers_newtonian_inspiral_rate():
    nu = torch.tensor(.19, dtype=DTYPE)
    ham = Nonspinning3PNHamiltonian(nu)
    rr = FactorizedRadiationReaction("din2009_nonspinning")
    # Newtonian circular energy and quadrupole luminosity imply
    # dr/d(t/M) = -64 nu/(5 r^3). Two radii resolve the first PN correction.
    relative_errors = []
    for radius in (1000., 2000.):
        omega = torch.tensor(radius**-1.5, dtype=DTYPE)
        r, pr, j = find_quasicircular_initial_conditions(ham, rr, omega, tol=1e-12)
        rate = ham.equations_of_motion_conservative(r, pr, j)[0]
        leading = -64*nu/(5*r**3)
        relative_errors.append((rate/leading-1).abs())
        assert pr < 0 and rate < 0
    assert relative_errors[0] < .01
    assert 1.7 < relative_errors[0]/relative_errors[1] < 2.3


def test_factorized_vector_field_shared_states_keep_parameter_batches_independent():
    nu = torch.tensor([.16, .24], dtype=DTYPE, requires_grad=True)
    s1 = torch.tensor([.4, -.2], dtype=DTYPE)
    s2 = torch.tensor([-.1, .3], dtype=DTYPE)
    rr = FactorizedRadiationReaction("pan2011_leading_spin")
    state = torch.tensor([14., .2, -.002, 4.3], dtype=DTYPE)

    def field(eta, a, b):
        ham = SEOBNRv4Hamiltonian(eta, a, b)
        # The public vector field accepts a state batch; expanding this state
        # mirrors a collection of independent identical initial coordinates.
        shaped = state.expand(eta.shape+(4,))
        return EOBTrajectoryIntegrator(ham, rr).vector_field(shaped)

    batch = field(nu, s1, s2)
    singles = torch.stack([field(nu[i], s1[i], s2[i]) for i in range(2)])
    torch.testing.assert_close(batch, singles, rtol=1e-12, atol=1e-15)
    batch_grad = torch.autograd.grad(batch.sum(), nu, retain_graph=True)[0]
    single_grad = torch.autograd.grad(singles.sum(), nu)[0]
    torch.testing.assert_close(batch_grad, single_grad, rtol=1e-11, atol=1e-14)
