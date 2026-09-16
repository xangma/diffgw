"""Primary-equation amplitude tests; no waveform oracle or implementation port."""

import math

import numpy as np
import pytest
import torch
from scipy.special import gamma

from torchwave.approximants.eob.factorized_modes import FactorizedModes
from torchwave.approximants.eob.seobnrv4_amplitude22 import (
    _rho22,
    seobnrv4_published_amplitude22 as amplitude22,
)


def t(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def inputs(v, x1=.8, chi1=.45, chi2=-.2):
    v = torch.as_tensor(v, dtype=torch.float64)
    nu = x1*(1-x1)
    heff = 1-v*v/2+(3/8+nu/6)*v**4
    return dict(omega=v**3, omega_zero_radial=v**3, heff_over_mu=heff,
                hreal_over_M=torch.sqrt(1+2*nu*(heff-1)),
                x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)


@pytest.mark.parametrize("x1,chi1,chi2", [(.5,.7,.7), (.8,.45,-.2), (.99,-.7,.3)])
def test_full_amplitude_has_independent_newtonian_1pn_and_spin_1p5pn_limits(x1, chi1, chi2):
    """Pan 1006.0431v2 (38a), real amplitude through v^3.

    Imaginary phase terms first affect the modulus at higher order.
    Weak-field Richardson limits isolate coefficients without fitting.
    """
    v = t([.01, .005, .0025, .00125])
    nu, delta = x1*(1-x1), 2*x1-1
    cs, ca = (chi1+chi2)/2, (chi1-chi2)/2
    hhat = amplitude22(**inputs(v,x1,chi1,chi2))/(8*math.sqrt(math.pi/5)*nu*v*v)
    c2 = -107/42+55*nu/42
    c3 = 2*math.pi-4/3*((1-nu)*cs+delta*ca)
    assert abs(float(hhat[-1])-1) < 2e-5
    assert abs(float(((hhat-1)/v**2)[-1])-c2) < .03
    residual = (hhat-1-c2*v*v)/v**3
    linear_cancel = 2*residual[1:]-residual[:-1]
    extrapolated = (4*linear_cancel[-1]-linear_cancel[-2])/3
    assert abs(float(extrapolated)-c3) < 3e-6


@pytest.mark.parametrize("x1", [.5, .8, .99])
def test_full_amplitude_2pn_spin_squared_from_buonanno_4p27b(x1):
    """For black holes S0+/M^2=S0-/M^2=X1*chi1+X2*chi2.

    Buonanno--Faye--Hinderer (3.4),(4.26),(4.27b) therefore gives a
    normalized h22 spin-squared coefficient (X1*chi1+X2*chi2)^2 v^4.
    """
    chi1, chi2 = .6, -.25
    nu, v = x1*(1-x1), t([.04,.02,.01])
    plus = amplitude22(**inputs(v,x1,chi1,chi2))
    minus = amplitude22(**inputs(v,x1,-chi1,-chi2))
    zero = amplitude22(**inputs(v,x1,0.,0.))
    coefficient = ((plus+minus)/2-zero)/(8*math.sqrt(math.pi/5)*nu*v**6)
    extrapolated = (4*coefficient[-1]-coefficient[-2])/3
    expected = (x1*chi1+(1-x1)*chi2)**2
    assert abs(float(extrapolated)-expected) < 3e-6


@pytest.mark.parametrize("x1", [.5,.8,1.])
def test_spin_derivative_isolates_bohe_2p5pn_and_3p5pn_coefficients(x1):
    """Isolate each published spin basis by differentiating at zero spin.

    After the Pan leading coefficient is removed the slope/v^5 is
    exactly c5+c7*v^2. Two evaluations determine these coefficients
    without introducing any numerical fit into the implementation.
    """
    nu, delta = x1*(1-x1), 2*x1-1
    v = t([.08,.16])
    for basis in ("s","a"):
        spin = t(0., requires_grad=True)
        zero = t(0.)
        slope = torch.stack([torch.autograd.grad(
            _rho22(vi,t(nu),t(delta),spin if basis == "s" else zero,
                   spin if basis == "a" else zero), spin, retain_graph=True)[0] for vi in v])
        c3 = -2/3*((1-nu) if basis == "s" else delta)
        remainder = (slope-c3*v**3)/v**5
        c7 = (remainder[1]-remainder[0])/(v[1]**2-v[0]**2)
        c5 = remainder[0]-c7*v[0]**2
        if basis == "s":
            expected5 = -34/21+49*nu/18+209*nu**2/126
            expected7 = 18733/15876+74749*nu/5292-245717*nu**2/63504+50803*nu**3/63504
        else:
            expected5 = -(34/21+19*nu/42)*delta
            expected7 = (18733/15876+50140*nu/3969+97865*nu**2/63504)*delta
        torch.testing.assert_close(c5,t(expected5),rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(c7,t(expected7),rtol=2e-11,atol=2e-11)


def test_remaining_kerr_spin_powers_and_whole_3p5pn_replacement():
    """Pan (29a), with exactly the v^5/v^7 replacements in Bohé.

    Even spin retains q^2 v^4/2, 89 q^2 v^6/252 and
    (18353 q^2/21168-q^4/8)v^8. Odd spin has no surviving q^3 v^7.
    """
    v, zero = t([.12,.3,.5]), t(0.)
    for q in [.2,.7]:
        plus = _rho22(v,zero,t(1.),t(q),zero)
        minus = _rho22(v,zero,t(1.),t(-q),zero)
        nonspin = _rho22(v,zero,t(1.),zero,zero)
        even = (plus+minus)/2-nonspin
        expected_even = q*q*v**4/2+89*q*q*v**6/252+(18353*q*q/21168-q**4/8)*v**8
        expected_odd = q*(-2*v**3/3-34*v**5/21+18733*v**7/15876)
        torch.testing.assert_close(even,expected_even,rtol=2e-10,atol=2e-16)
        torch.testing.assert_close((plus-minus)/2,expected_odd,rtol=3e-12,atol=2e-16)


def test_zero_radial_circular_limit_and_nonkeplerian_power():
    v = t([.08,.2,.4])
    args = inputs(v,chi1=0.,chi2=0.)
    state = torch.stack((v**-2,torch.zeros_like(v),torch.zeros_like(v),v**-1),-1)
    circular = FactorizedModes(lmax=2).amplitudes(
        state,args["omega"],args["heff_over_mu"],args["hreal_over_M"],args["x1"],args["x2"])[2,2]
    # A2 gives vPhi=vOmega exactly if pr=0, even without imposing dH/dr=0.
    torch.testing.assert_close(amplitude22(**args),circular,rtol=5e-15,atol=0.)
    changed = dict(args,omega_zero_radial=2*args["omega_zero_radial"])
    torch.testing.assert_close(amplitude22(**changed)/amplitude22(**args),
                               torch.full_like(v,2**(-4/3)),rtol=4e-15,atol=0.)


def test_tail_uses_total_energy_and_matches_independent_complex_gamma():
    v = t([.01,.2,.5])
    args = inputs(v)
    args.update(omega_zero_radial=args["omega"]*1.2,heff_over_mu=t([.99,.97,.91]),
                hreal_over_M=t([1.,.99,.96]))
    nu, delta = args["x1"]*args["x2"], args["x1"]-args["x2"]
    cs, ca = (args["chi1"]+args["chi2"])/2, (args["chi1"]-args["chi2"])/2
    rho = _rho22(v,t(nu),t(delta),t(cs),t(ca)).numpy()
    khat = (2*args["omega"]*args["hreal_over_M"]).numpy()
    tail = np.abs(gamma(3-2j*khat)/2*np.exp(math.pi*khat))
    vphi = (args["omega"]*args["omega_zero_radial"]**(-2/3)).numpy()
    expected = 8*math.sqrt(math.pi/5)*nu*vphi**2*args["heff_over_mu"].numpy()*tail*rho**2
    np.testing.assert_allclose(amplitude22(**args).numpy(),expected,rtol=2e-14,atol=0.)


@pytest.mark.parametrize("x1", [.5,.82])
def test_gradients_of_frequencies_energies_masses_and_individual_spins(x1):
    parameters = t([.02,.021,.97,.992,x1,.45,-.2],requires_grad=True)

    def f(p):
        return amplitude22(p[0],omega_zero_radial=p[1],heff_over_mu=p[2],
                           hreal_over_M=p[3],x1=p[4],x2=1-p[4],chi1=p[5],chi2=p[6])

    assert torch.autograd.gradcheck(f,(parameters,),eps=1e-6,atol=2e-7,rtol=2e-5)
    gradient = torch.autograd.grad(f(parameters),parameters)[0]
    assert torch.all(torch.isfinite(gradient)) and torch.all(gradient != 0)


def test_broadcast_exchange_endpoints_and_float32_inference():
    v, x1 = t([.1,.2,.4]), t([[.5],[.8]])
    args = inputs(v,x1)
    actual = amplitude22(**args)
    assert actual.shape == (2,3)
    exchanged = dict(args,x1=args["x2"],x2=args["x1"],chi1=args["chi2"],chi2=args["chi1"])
    torch.testing.assert_close(actual,amplitude22(**exchanged),rtol=2e-15,atol=0.)
    for row,x in enumerate([.5,.8]):
        torch.testing.assert_close(actual[row],amplitude22(**inputs(v,x)))
    assert torch.all(amplitude22(**inputs(v,1.)) == 0)
    with torch.inference_mode():
        single = {key:value.float() if isinstance(value,torch.Tensor) else value for key,value in args.items()}
        torch.testing.assert_close(amplitude22(**single),actual.float(),rtol=2e-6,atol=0.)


@pytest.mark.parametrize("replacement", [dict(omega_zero_radial=0.),dict(hreal_over_M=-.01),
                                            dict(x1=.9),dict(chi1=1.01),dict(heff_over_mu=float("nan"))])
def test_invalid_inputs_are_rejected(replacement):
    with pytest.raises(ValueError):
        amplitude22(**dict(inputs(.2),**replacement))
