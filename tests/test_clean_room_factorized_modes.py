"""Independent primary-equation limits; no waveform oracle or port is imported."""

import math

import numpy as np
import pytest
import torch
from scipy.special import gamma

from torchwave.approximants.eob.factorized_modes import (
    FactorizedModes,
    _tail_abs_squared,
    _tail_phase,
    energy_flux_from_modes,
)


def t(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def circular_inputs(v, x1=.7, phi=.31):
    v = torch.as_tensor(v, dtype=torch.float64)
    nu = x1*(1-x1)
    x = v*v
    # DIN (17)--(18), plus the 2PN effective-energy coefficient obtained
    # from the circular binding energy and DIN energy map (14).
    heff = 1-x/2+(3/8+nu/6)*x*x
    hreal = torch.sqrt(1+2*nu*(heff-1))
    j = (1+(1.5+nu/6)*x)/v
    state = torch.stack((1/x, torch.full_like(x, phi), torch.zeros_like(x), j), -1)
    return state, v**3, heff, hreal, x1, 1-x1


def test_newtonian_modes_match_independently_published_strains():
    """Pan 1006.0431v2 (38a)--(38e), leading terms including complex signs."""
    v, phi, x1, x2 = 1e-6, .37, .8, .2
    nu, d = x1*x2, x1-x2
    state = t([v**-2, phi, 0., 1/v])
    actual = FactorizedModes(3)(state, v**3, 1., 1., x1, x2)
    coefficients = {
        (2, 2): -8*math.sqrt(math.pi/5)*nu*v**2,
        (2, 1): -8j/3*math.sqrt(math.pi/5)*nu*d*v**3,
        (3, 3): 3j*math.sqrt(6*math.pi/7)*nu*d*v**3,
        (3, 2): -8/3*math.sqrt(math.pi/7)*nu*(1-3*nu)*v**4,
        (3, 1): -1j/3*math.sqrt(2*math.pi/35)*nu*d*v**3,
    }
    for (ell, m), coefficient in coefficients.items():
        expected = coefficient*np.exp(-1j*m*phi)
        torch.testing.assert_close(actual[ell, m], torch.tensor(expected), rtol=1e-10, atol=0.)


@pytest.mark.parametrize("x1", [.5, .8, .99])
def test_total_flux_recovers_newtonian_1pn_tail_and_2pn_coefficients(x1):
    """Buonanno--Damour gr-qc/0001013v2 (3.31)--(3.33), through 2PN.

    Successive weak-field limits isolate coefficients of the *computed*
    flux. Richardson extrapolation cancels higher-order remainders; it
    does not fit, modify, or feed back into the mode model.
    """
    nu = x1*(1-x1)
    v = t([.02, .01, .005, .0025])
    fhat = FactorizedModes().flux(*circular_inputs(v, x1))/(32/5*nu**2*v**10)
    f2 = -1247/336-35*nu/12
    f3 = 4*math.pi
    f4 = -44711/9072+9271*nu/504+65*nu**2/18
    assert abs(float(fhat[-1])-1) < 3e-5
    assert abs(float(((fhat-1)/v**2)[-1])-f2) < .04
    assert abs(float(((fhat-1-f2*v**2)/v**3)[-1])-f3) < .02
    c4 = (fhat-1-f2*v**2-f3*v**3)/v**4
    first = 2*c4[1:]-c4[:-1]
    extrapolated = (4*first[-1]-first[-2])/3
    assert abs(float(extrapolated)-f4) < 2e-4


@pytest.mark.parametrize("ell", range(2, 9))
def test_tail_matches_independent_complex_gamma_mathematics(ell):
    omega, hreal = t([1e-12, 1e-5, .03, .2]), t([1., .99, .96, 1.02])
    m = ell
    khat = m*omega*hreal
    expected = (gamma(ell+1-2j*khat.numpy())/math.factorial(ell)
                * np.exp(math.pi*khat.numpy()+2j*khat.numpy()*np.log(4*m*omega.numpy())))
    actual = torch.polar(_tail_abs_squared(ell, khat).sqrt(), _tail_phase(ell, m, omega, hreal))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-14, atol=2e-14)


def test_all_35_amplitudes_and_flux_use_the_same_complex_modes():
    model = FactorizedModes(residual_phase="zero-unpublished")
    args = circular_inputs(t([.1, .25, .4]))
    modes = model(*args)
    amplitudes = model.amplitudes(*args)
    assert len(modes) == len(amplitudes) == 35
    for mode in modes:
        torch.testing.assert_close(amplitudes[mode], modes[mode].abs(), rtol=3e-15, atol=0.)
    expected = sum((m*args[1])**2*h.abs()**2 for (_, m), h in modes.items())/(8*math.pi)
    torch.testing.assert_close(model.flux(*args), expected, rtol=3e-15, atol=0.)
    torch.testing.assert_close(energy_flux_from_modes(modes, args[1]), expected)


def test_selected_modes_preserve_phase_policy_and_negative_m_symmetry():
    model = FactorizedModes()
    args = circular_inputs(t([.1, .3]))
    all_modes = FactorizedModes(residual_phase="zero-unpublished")(*args)
    for ell, m in [(2, 2), (3, 1), (5, 5)]:
        h = model.mode(ell, m, *args)
        torch.testing.assert_close(h, all_modes[ell, m])
        torch.testing.assert_close(model.mode(ell, -m, *args), (-1)**ell*h.conj())
    with pytest.raises(NotImplementedError, match="delta_lm"):
        model.mode(8, 8, *args)
    with pytest.raises(ValueError, match="Requested mode"):
        model.mode(2, 0, *args)
    permissive = FactorizedModes(residual_phase="zero-unpublished")
    torch.testing.assert_close(permissive.mode(8, 8, *args), all_modes[8, 8])


def test_mass_exchange_and_equal_mass_odd_modes_all_35():
    model = FactorizedModes(residual_phase="zero-unpublished")
    state, omega, heff, hreal, x1, x2 = circular_inputs(.25, .73)
    first = model(state, omega, heff, hreal, x1, x2)
    exchanged = model(state, omega, heff, hreal, x2, x1)
    equal = model(state, omega, heff, hreal, .5, .5)
    for (ell, m), value in first.items():
        torch.testing.assert_close(exchanged[ell, m], (-1)**m*value, rtol=3e-15, atol=0.)
        if m % 2:
            assert equal[ell, m] == 0


def test_leading_spin_exchange_and_equal_mass_unequal_spin_limits():
    """Pan (38b,d): spin-induced 21 and 32 Newtonian terms at equal masses."""
    model = FactorizedModes(4, spin_model="pan2011-leading-f")
    state, omega, heff, hreal, x1, x2 = circular_inputs(.2, .69)
    first = model(state, omega, heff, hreal, x1, x2, chi1=.7, chi2=-.3)
    exchanged = model(state, omega, heff, hreal, x2, x1, chi1=-.3, chi2=.7)
    for (ell, m), value in first.items():
        torch.testing.assert_close(exchanged[ell, m], (-1)**m*value, rtol=3e-15, atol=1e-18)

    v, phi, nu = 1e-5, .2, .25
    state = t([v**-2, phi, 0., 1/v])
    chi1, chi2 = .8, -.2
    spin = model(state, v**3, 1., 1., .5, .5, chi1=chi1, chi2=chi2)
    zero = model(state, v**3, 1., 1., .5, .5)
    # The apparently singular delta_m factor cancels in the physical 21 mode.
    expected21 = 4j*math.sqrt(math.pi/5)*nu*((chi1-chi2)/2)*v**4*np.exp(-1j*phi)
    expected32 = -32/3*math.sqrt(math.pi/7)*nu**2*((chi1+chi2)/2)*v**5*np.exp(-2j*phi)
    np.testing.assert_allclose(spin[2, 1].numpy(), expected21, rtol=1e-10, atol=0.)
    np.testing.assert_allclose((spin[3, 2]-zero[3, 2]).numpy(), expected32, rtol=1e-9, atol=0.)
    for m in (1, 3):
        assert torch.isfinite(spin[4, m]) and spin[4, m].abs() > 0


def test_gradients_include_state_frequency_energies_masses_and_spins():
    model = FactorizedModes(4, spin_model="pan2011-leading-f")
    state = t([12., .4, -.02, 3.8], requires_grad=True)
    args = (state, t(.027, requires_grad=True), t(.96, requires_grad=True),
            t(.99, requires_grad=True), t(.5, requires_grad=True),
            t(.7, requires_grad=True), t(-.3, requires_grad=True))

    def evaluate(state, omega, heff, hreal, x1, chi1, chi2):
        modes = model(state, omega, heff, hreal, x1, 1-x1, chi1=chi1, chi2=chi2)
        return torch.stack(list(modes.values()))

    assert torch.autograd.gradcheck(evaluate, args, atol=2e-7, rtol=2e-5)
    assert torch.autograd.gradcheck(
        lambda s, o, e, h, q, a, b: model.flux(s, o, e, h, q, 1-q, chi1=a, chi2=b),
        args, atol=1e-9, rtol=2e-5,
    )


def test_high_mode_flux_gradients_batching_and_inference():
    model = FactorizedModes()
    args = circular_inputs(t([.17, .29]), .67)
    batched = model.flux(*args)
    scalar = torch.stack([model.flux(*(arg[i] if isinstance(arg, torch.Tensor) else arg
                                      for arg in args)) for i in range(2)])
    torch.testing.assert_close(batched, scalar)
    with torch.inference_mode():
        torch.testing.assert_close(model.flux(*args), batched)
    q, v = t(.63, requires_grad=True), t(.25, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, q: model.flux(*circular_inputs(v, q)), (v, q),
                                    atol=1e-9, rtol=2e-5)


@pytest.mark.parametrize("real_components", [False, True])
def test_equal_mass_odd_mode_flux_hessian_has_the_primary_newtonian_limit(real_components):
    """Pan (38b,c): h21 and h33 vanish but their flux curvature does not."""
    delta, v = t(0., requires_grad=True), 1e-4
    state = t([v**-2,.37,0.,1/v])
    all_modes = FactorizedModes(3)(state,v**3,1.,1.,(1+delta)/2,(1-delta)/2)
    odd = {mode: all_modes[mode] for mode in ((2,1),(3,3))}
    assert all(value == 0 for value in odd.values())
    if real_components:
        flux = energy_flux_from_modes({mode: h.real for mode,h in odd.items()},t(v**3))
        flux = flux+energy_flux_from_modes({mode: h.imag for mode,h in odd.items()},t(v**3))
    else:
        flux = energy_flux_from_modes(odd, t(v**3))
    first = torch.autograd.grad(flux,delta,create_graph=True)[0]
    second = torch.autograd.grad(first,delta)[0]
    # The two independent Newtonian strain coefficients yield 1/45 and
    # 243/224 in d²(flux)/d(delta)²/v**12, including nu(delta=0)=1/4.
    torch.testing.assert_close(second/v**12,t(1/45+243/224),rtol=2e-7,atol=0)


def test_public_factorized_flux_hessian_is_smooth_at_equal_masses():
    x1, v = t(.5,requires_grad=True), t(.3,requires_grad=True)
    flux = lambda v, x1: FactorizedModes().flux(*circular_inputs(v,x1))
    assert torch.autograd.gradgradcheck(flux,(v,x1),grad_outputs=t(1.),atol=1e-10,rtol=2e-5)


def test_unsupported_physics_and_invalid_inputs_are_explicit():
    args = circular_inputs(.2)
    with pytest.raises(NotImplementedError, match="delta_lm"):
        FactorizedModes()(*args)
    with pytest.raises(NotImplementedError, match="full v4"):
        FactorizedModes(spin_model="seobnrv4")
    with pytest.raises(NotImplementedError, match="ell=4"):
        FactorizedModes(spin_model="pan2011-leading-f")
    with pytest.raises(NotImplementedError, match="Nonzero spins"):
        FactorizedModes().flux(*args, chi1=.1)
    with pytest.raises(ValueError, match="positive"):
        FactorizedModes().flux(args[0], 0., *args[2:])
    with pytest.raises(ValueError, match="sum to one"):
        FactorizedModes().flux(*args[:4], .6, .6)
    with pytest.raises(ValueError, match="positive-m"):
        energy_flux_from_modes({(2, -2): t(1.)}, t(.02))
