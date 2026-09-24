"""Primary-equation limits and numerical semantics, without waveform oracles."""
import math

import numpy as np
import pytest
import torch
from scipy.special import gamma

from torchwave.approximants.eob.factorized_modes import _newtonian_constant
from torchwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


def t(value, **kw):
    return torch.tensor(value, dtype=torch.float64, **kw)


def model(**kw):
    return SEOBNRv4PublishedFlux(spin_truncation="tabulated", **kw)


def inputs(v=.2, x1=.8, chi1=.6, chi2=-.25):
    v = torch.as_tensor(v, dtype=torch.float64)
    state = torch.stack((v**-2, torch.zeros_like(v), -.02*torch.ones_like(v), v**-1), -1)
    return state, dict(omega=v**3, omega_zero_radial=1.03*v**3,
                      heff_over_mu=.98, hreal_over_M=.995,
                      x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)


def test_explicit_truncation_is_required_and_changes_only_published_ell8_even_modes():
    with pytest.raises(TypeError):
        SEOBNRv4PublishedFlux()
    with pytest.raises(NotImplementedError):
        SEOBNRv4PublishedFlux(spin_truncation="automatic")
    state, args = inputs()
    tabulated = model().amplitudes(state, **args)
    generic = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").amplitudes(state, **args)
    assert len(tabulated) == 35
    for mode in tabulated:
        if mode[0] == 8 and mode[1] % 2 == 0:
            assert generic[mode] < tabulated[mode]
        else:
            torch.testing.assert_close(generic[mode], tabulated[mode], rtol=0, atol=0)


def test_dominant_mode_reuses_reviewed_bohe_amplitude_without_nqc():
    state, args = inputs(t([.08,.2,.35]), x1=t([[.5],[.82]]))
    torch.testing.assert_close(model().amplitudes(state, **args)[2,2],
                               seobnrv4_published_amplitude22(**args), rtol=7e-15, atol=0)


def test_all_modes_have_newtonian_limit_and_exact_nonkeplerian_power():
    state, args = inputs(1e-5, chi1=0, chi2=0)
    args.update(omega_zero_radial=args["omega"], heff_over_mu=1, hreal_over_M=1)
    m = model()
    actual = m.amplitudes(state, **args)
    changed = m.amplitudes(state, **dict(args, omega_zero_radial=2*args["omega_zero_radial"]))
    for (ell, order), amp in actual.items():
        epsilon = (ell+order) % 2
        p = ell+epsilon-1
        c = .2**p+(-1)**order*.8**p
        expected = abs(_newtonian_constant(ell,order)*c)*.16*1e-5**(ell+epsilon)
        assert abs(float(amp)/expected-1) < 3e-9
        torch.testing.assert_close(changed[ell,order]/amp, t(2**(-2*(ell+epsilon)/3)),
                                   rtol=1e-14, atol=0)
    torch.testing.assert_close(m.energy_flux(state, **args)/(.16**2*1e-5**10),
                               t(32/5), rtol=1e-9, atol=0)


def test_all_tail_moduli_match_independent_complex_gamma_energy_ratio():
    state, args = inputs(t([.08,.25,.45]))
    m = model()
    a = m.amplitudes(state, **args)
    b = m.amplitudes(state, **dict(args, hreal_over_M=1.1))
    for (ell, order), amp in a.items():
        khat = order*args["omega"].numpy()
        def tail(k):
            return np.abs(gamma(ell+1-2j*k)*np.exp(math.pi*k))
        np.testing.assert_allclose((b[ell,order]/amp).numpy(),
                                   tail(1.1*khat)/tail(.995*khat), rtol=2e-14)


@pytest.mark.parametrize("x1", [.65,.85,.99])
def test_bfh_2pn_h21_spin_coefficient_after_independent_pn_angular_momentum(x1):
    """BFH (4.27a) gives h21 spin v**3, not a residual-f coefficient.

    In Pan's phase convention its normalized coefficient is
    ((-7+33nu)chiS+(-7+205nu)chiA/delta)/14. Insert Pan (40)'s J
    and remove the independent Gamma tail before extracting that order.
    """
    nu, delta = x1*(1-x1), 2*x1-1
    v = t([.008,.004,.002])
    for basis in ("s", "a"):
        spin = t(0., requires_grad=True)
        cs, ca = (spin, t(0.)) if basis == "s" else (t(0.), spin)
        j = (1+(9+nu)/6*v*v-10/3*((1-nu/2)*cs+delta*ca)*v**3)/v
        state = torch.stack((v**-2, v*0, v*0, j), -1)
        args = dict(omega=v**3, omega_zero_radial=v**3, heff_over_mu=1, hreal_over_M=1,
                    x1=x1, x2=1-x1, chi1=cs+ca, chi2=cs-ca)
        amp = model(lmax=2).amplitudes(state, **args)[2,1]
        khat = v.numpy()**3
        tail = t(np.abs(gamma(3-2j*khat)/2*np.exp(math.pi*khat)))
        normalized = amp/(8/3*math.sqrt(math.pi/5)*nu*delta*v**3*tail)
        slope = torch.stack([torch.autograd.grad(value, spin, retain_graph=True)[0]
                             for value in normalized])
        leading = -1.5*(1 if basis == "s" else 1/delta)
        coefficient = (slope-leading*v)/v**3
        extrapolated = (4*coefficient[-1]-coefficient[-2])/3
        expected = (-7+33*nu)/14 if basis == "s" else (-7+205*nu)/(14*delta)
        torch.testing.assert_close(extrapolated, t(expected), rtol=3e-6, atol=3e-6)


def test_pan_f26_removes_unphysical_1pn_spin_square_in_current32():
    v = t(1e-4)
    state, args = inputs(v, x1=.6)
    spin = t(0., requires_grad=True)
    args.update(chi1=spin, chi2=spin, omega_zero_radial=v**3)
    amp = model(lmax=3).amplitudes(state, **args)[3,2]
    zero_amp = amp.detach()
    first = torch.autograd.grad(amp, spin, create_graph=True)[0]
    second = torch.autograd.grad(first, spin)[0]
    assert abs(float(second/(zero_amp*v*v))) < 2e-7


@pytest.mark.parametrize("flux_choices", [
    {},
    dict(mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
         mode_32_spin_square="lal_v4_zero", mode_33_higher_spin="lal_v4_rho"),
])
def test_equal_mass_regularization_exchange_continuity_and_test_mass_endpoint(flux_choices):
    m = model(**flux_choices)
    for x1 in (.5, .5-1e-9, .5+1e-9, .8):
        state, args = inputs(x1=x1)
        a = m.amplitudes(state, **args)
        b = m.amplitudes(state, **dict(args, x1=1-x1, x2=x1, chi1=args["chi2"], chi2=args["chi1"]))
        for mode in a:
            torch.testing.assert_close(a[mode], b[mode], rtol=3e-14, atol=0)
            assert torch.isfinite(a[mode])
        if x1 == .5:
            equal = a
            assert a[2,1] > 0
        if 0 < abs(x1-.5) < 2e-9:
            for mode in ((2,1),(3,3),(3,1),(4,3),(4,1)):
                torch.testing.assert_close(a[mode], equal[mode], rtol=3e-6, atol=1e-12)
    state, args = inputs(x1=.5, chi1=.7, chi2=.7)
    assert all(value == 0 for (ell, order), value in m.amplitudes(state, **args).items() if order % 2)
    state, args = inputs(x1=1.)
    assert m.energy_flux(state, **args) == 0


@pytest.mark.parametrize("x1", [.5,.8])
@pytest.mark.parametrize("flux_choices", [
    {},
    dict(mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
         mode_32_spin_square="lal_v4_zero", mode_33_higher_spin="lal_v4_rho"),
])
def test_flux_gradients_include_mass_spins_frequencies_energies_and_momentum(x1, flux_choices):
    p = t([.02,.021,.98,.995,x1,.6,-.25,4.], requires_grad=True)
    flux = model(**flux_choices)
    def f(p):
        state = torch.stack((p[7]*0+12,p[7]*0,p[7]*0-.02,p[7]))
        return flux.energy_flux(state, omega=p[0], omega_zero_radial=p[1],
                                heff_over_mu=p[2], hreal_over_M=p[3],
                                x1=p[4], x2=1-p[4], chi1=p[5], chi2=p[6])
    assert torch.autograd.gradcheck(f, (p,), atol=1e-9, rtol=3e-5)
    gradient = torch.autograd.grad(f(p), p)[0]
    assert torch.isfinite(gradient).all() and (gradient != 0).all()


@pytest.mark.parametrize("context", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_hamiltonian_adapter_batched_trajectories_forces_and_inference_created_parameters(context):
    with context():
        ham = SEOBNRv4Hamiltonian(t([.24,.16]), t([.4,-.2]), t([-.1,.3]), delta=t([.2,.6]))
        r = t([[15.,13.,11.],[16.,14.,12.]])
        pr, j = torch.full_like(r,-.02), torch.sqrt(r)+.6
        state = torch.stack((r,r*0,pr,j),-1)
        m = model()
        args = m.from_hamiltonian(state, ham)
        actual = m.energy_flux(state, **args)
        # Supplied frequency must survive the adapter; only omega0 is recomputed.
        supplied = args["omega"]*1.01
        fr, torque = m.radiation_reaction_forces(supplied,pr,j,r=r,hamiltonian=ham)
        expected = m.energy_flux(state, **dict(args,omega=supplied))
        torch.testing.assert_close(-torque*ham.nu[:,None]*supplied, expected)
        torch.testing.assert_close(fr,pr/j*torque)
        assert actual.shape == (2,3) and torch.isfinite(actual).all()
        for row in range(2):
            single = SEOBNRv4Hamiltonian(ham.nu[row],ham.chi1[row],ham.chi2[row],delta=ham.delta[row])
            single_args = m.from_hamiltonian(state[row], single)
            torch.testing.assert_close(actual[row],m.energy_flux(state[row], **single_args))


@pytest.mark.parametrize("delta", [0., .6])
def test_hamiltonian_adapter_preserves_first_and_second_parameter_derivatives(delta):
    p = t([delta,.6,-.25,12.,-.02,4.], requires_grad=True)
    def flux(p):
        ham = SEOBNRv4Hamiltonian((1-p[0]**2)/4,p[1],p[2],delta=p[0])
        state = torch.stack((p[3],p[3]*0,p[4],p[5]))
        return model().energy_flux(state, **model().from_hamiltonian(state,ham))
    assert torch.autograd.gradcheck(flux, (p,), atol=1e-10, rtol=1e-5)
    assert torch.autograd.gradgradcheck(flux, (p,), grad_outputs=t(1.), atol=1e-10, rtol=1e-5)


def test_float32_preserves_dtype_and_agrees_with_float64():
    state, args = inputs(t([.12,.2,.3]), x1=.7)
    actual = model().energy_flux(state.float(), **args)
    expected = model().energy_flux(state, **args)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("bad", [dict(omega=0),dict(omega_zero_radial=-1),dict(heff_over_mu=0),
                                 dict(hreal_over_M=float("nan")),dict(x1=.99),dict(chi1=1.1)])
def test_input_errors(bad):
    state, args = inputs()
    with pytest.raises(ValueError):
        model().energy_flux(state, **dict(args, **bad))
