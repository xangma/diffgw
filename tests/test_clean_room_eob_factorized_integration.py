"""Independent energy-balance and covariance checks for factorized EOB wiring."""

import pytest
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.radiation_reaction import FactorizedRadiationReaction
from torchwave.waveforms import get_fd_waveform


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
    ham = EOBHamiltonian(nu, nu*0, nu*0)
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


@pytest.mark.parametrize("model,spin", [("din2009_nonspinning", 0.), ("pan2011_leading_spin", .3)])
def test_factorized_waveform_preserves_total_mass_distance_and_time_scaling(model, spin):
    f = torch.tensor([30., 50., 70., 90.], dtype=DTYPE)
    args = dict(approximant="SEOBNR_ODE", conservative_model="seobnrv4_published",
                radiation_model=model, chi1=spin, chi2=-spin/2, inclination=.8,
                num_steps=3, dt_geom=1., phic=.2)
    first = get_fd_waveform(mass1=30., mass2=20., distance=200., tc=.002,
                            sample_frequencies=f, f_lower=30., **args)
    second = get_fd_waveform(mass1=60., mass2=40., distance=600., tc=.004,
                             sample_frequencies=f/2, f_lower=15., **args)
    for a, b in zip(first, second):
        torch.testing.assert_close(b, 4*a/3, rtol=3e-11, atol=1e-33)


def test_factorized_waveform_batched_spins_and_inference_match_scalar_calls():
    f = torch.tensor([30., 45., 60.], dtype=DTYPE)
    m1 = torch.tensor([30., 35.], dtype=DTYPE)
    spin = torch.tensor([.3, -.2], dtype=DTYPE)
    args = dict(approximant="SEOBNR_ODE", mass2=20., sample_frequencies=f,
                f_lower=30., conservative_model="seobnrv4_published",
                radiation_model="pan2011_leading_spin", num_steps=2, dt_geom=1.)
    batch = get_fd_waveform(mass1=m1, chi1=spin, **args)
    with torch.inference_mode():
        inference = get_fd_waveform(mass1=m1.clone(), chi1=spin.clone(), **args)
    for b, value in zip(batch, inference):
        torch.testing.assert_close(b, value, rtol=3e-12, atol=1e-34)
    for i in range(2):
        single = get_fd_waveform(mass1=m1[i], chi1=spin[i], **args)
        for b, value in zip(batch, single):
            torch.testing.assert_close(b[i], value, rtol=3e-11, atol=1e-33)


def test_nonspinning_radiation_rejects_spins_instead_of_silently_dropping_them():
    with pytest.raises(NotImplementedError, match="Nonzero spins"):
        get_fd_waveform("SEOBNR_ODE", mass1=30., mass2=20., chi1=.1,
                        sample_frequencies=torch.tensor([30., 40.], dtype=DTYPE),
                        f_lower=30., radiation_model="din2009_nonspinning", num_steps=2)
