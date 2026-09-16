"""Independent special-function, PN-limit and mode-geometry checks."""

import math

import numpy as np
import pytest
import torch
from scipy.special import digamma, loggamma

from torchwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22
from torchwave.approximants.eob.seobnrv4_mode22 import (
    delta22_bohe_spin,
    delta22_nonspin,
    delta22_pan_spin,
    delta22_positive_chi_calibration,
    seobnrv4_published_delta22 as delta22,
    seobnrv4_published_mode22 as mode22,
    seobnrv4_published_phase22 as phase22,
    tail22,
    tail22_phase,
)


def t(value, **kwargs):
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def phase_inputs(**kwargs):
    result = dict(hreal_over_M=.98, x1=.7, x2=.3, chi1=.6, chi2=-.2,
                  spin_v6="bohe_total_energy", include_pan_v8=True,
                  include_positive_chi_540=False)
    result.update(kwargs)
    return result


def mode_inputs(**kwargs):
    result = dict(omega_zero_radial=.03, heff_over_mu=.9, hreal_over_M=.98,
                  x1=.7, x2=.3, chi1=.6, chi2=-.2)
    result.update(kwargs)
    return result


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 4e-14), (torch.float32, 2e-5)])
def test_complex_tail_and_continuous_phase_against_scipy_gamma(dtype, tolerance):
    omega = torch.logspace(-12, math.log10(2), 61, dtype=dtype)[:, None]
    energy = torch.tensor([.8, 1., 1.2], dtype=dtype)[None, :]
    w, e = omega.double().numpy(), energy.double().numpy()
    khat = 2*w*e
    logarithm = loggamma(3-2j*khat)-math.log(2)+math.pi*khat
    logarithm += 2j*khat*np.log(8*w/math.sqrt(math.e))
    np.testing.assert_allclose(tail22_phase(omega, hreal_over_M=energy),
                               logarithm.imag, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(tail22(omega, hreal_over_M=energy), np.exp(logarithm),
                               atol=tolerance, rtol=tolerance)
    # The phase is genuinely continuous, not the principal argument of T.
    assert float(tail22_phase(omega[-1], hreal_over_M=energy).max()) > math.pi


def test_tail_derivatives_against_independent_digamma():
    omega = t([1e-9, .003, .09, .6], requires_grad=True)
    energy = t([1., .97, 1.1, .85], requires_grad=True)
    result = tail22_phase(omega, hreal_over_M=energy)
    dw, de = torch.autograd.grad(result.sum(), (omega, energy))
    w, e = omega.detach().numpy(), energy.detach().numpy()
    psi = np.real(digamma(3-4j*w*e))
    logarithm = np.log(8*w/math.sqrt(math.e))
    np.testing.assert_allclose(dw, 4*e*(logarithm+1-psi), atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(de, 4*w*(logarithm-psi), atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("x1,chi1,chi2", [(.5, 0., 0.), (.7, .6, -.3), (.95, -.8, .4)])
def test_full_mode_newtonian_sign_and_pan_1p5pn_imaginary_strain(x1, chi1, chi2):
    # Pan (21),(38a) independently specify the imaginary full-strain term
    # 12*v^3*log(v/v0), including the constant that depends on tail radius.
    v = t([.006, .003, .0015])
    nu = x1*(1-x1)
    heff = 1-v*v/2+(3/8+nu/6)*v**4
    energy = torch.sqrt(1+2*nu*(heff-1))
    d = delta22(v**3, **phase_inputs(hreal_over_M=energy, x1=x1, x2=1-x1,
                                    chi1=chi1, chi2=chi2))
    phi = .37
    h = mode22(v**3, phi, residual_phase=d, **mode_inputs(
        omega_zero_radial=v**3, heff_over_mu=heff, hreal_over_M=energy,
        x1=x1, x2=1-x1, chi1=chi1, chi2=chi2))
    newtonian = -8*math.sqrt(math.pi/5)*nu*v**2*np.exp(-2j*phi)
    normalized = h/newtonian
    log_v0 = 11/36-np.euler_gamma/3-2*math.log(2)/3-math.log(2/math.sqrt(math.e))/3
    imaginary_residual = normalized.imag/v**3-12*(torch.log(v)-log_v0)
    assert abs(complex(normalized[-1])-1) < 8e-6
    assert abs(float(imaginary_residual[-1])) < .001
    assert abs(float(imaginary_residual[-1])) < abs(float(imaginary_residual[0]))/8


def test_nonspin_high_order_term_matches_pan_test_mass_phase():
    # At nu=0, total Hreal/M=1, A16 must recover Pan (27a)'s nonspin
    # Kerr-limit phase. Isolate the independent printed v^9 coefficient.
    v = t([.16, .24, .35])
    residual = delta22_nonspin(v**3, hreal_over_M=1., nu=0.)
    isolated = (residual-7*v**3/3-428*math.pi*v**6/105)/v**9
    torch.testing.assert_close(isolated, torch.full_like(v, 1712*math.pi**2/315-2203/81),
                               atol=2e-10, rtol=2e-12)


def test_bohe_energy_repair_has_the_stated_pn_order_and_kerr_limit():
    v = t([.1, .05, .025])
    nu, q = .21, .6
    total_energy = torch.sqrt(1-nu*v*v)
    correction = delta22_bohe_spin(v**3, hreal_over_M=total_energy, q_kerr=q)
    coefficient = correction/v**6
    assert abs(float(coefficient[-1])+4*q/3) < .0002
    # Replacing total energy by binding energy would instead scale v^10.
    assert abs(float(correction[0]/correction[1])/64-1) < .002
    bohe = delta22_bohe_spin(v**3, hreal_over_M=1., q_kerr=q)
    pan = delta22_pan_spin(v**3, q_kerr=q, include_v6=True, include_v8=False)
    torch.testing.assert_close(bohe, pan)


@pytest.mark.parametrize("spin_v6", ["none", "bohe_total_energy", "pan_test_particle", "bohe_plus_pan"])
def test_required_choices_remain_independent_of_each_other(spin_v6):
    omega = t([.005, .04])
    base = delta22(omega, **phase_inputs(spin_v6=spin_v6, include_pan_v8=False))
    plus8 = delta22(omega, **phase_inputs(spin_v6=spin_v6))
    plus540 = delta22(omega, **phase_inputs(spin_v6=spin_v6, include_pan_v8=False,
                                         include_positive_chi_540=True))
    q = .7**2*.6+.3**2*(-.2)
    torch.testing.assert_close((plus8-base)/omega.pow(8/3), torch.full_like(omega, 20*q/63))
    torch.testing.assert_close((plus540-base)/omega**2,
                               torch.full_like(omega, 540*.21*q/(1-2*.21)))


def test_deliberate_sum_is_available_without_selecting_an_inheritance_default():
    omega = t([.003, .08])
    phase = {choice: delta22(omega, **phase_inputs(spin_v6=choice)) for choice in
             ("none", "bohe_total_energy", "pan_test_particle", "bohe_plus_pan")}
    torch.testing.assert_close(phase["bohe_plus_pan"]-phase["none"],
                               phase["bohe_total_energy"]+phase["pan_test_particle"]-2*phase["none"])
    for choice in ("spin_v6", "include_pan_v8", "include_positive_chi_540"):
        kwargs = phase_inputs()
        del kwargs[choice]
        with pytest.raises(TypeError):
            delta22(omega, **kwargs)
    with pytest.raises(TypeError):
        mode22(omega, .2, **mode_inputs())


def test_positive_chi_calibration_piecewise_domain_and_derivative():
    chi = t([-.4, 0., .4], requires_grad=True)
    result = delta22_positive_chi_calibration(t(.02), nu=.25, chi=chi)
    assert result[0] == result[1] == 0
    derivative, = torch.autograd.grad(result.sum(), (chi,))
    torch.testing.assert_close(derivative, t([0., .054, .054]))


def test_continuous_gw_phase_derivative_matches_complex_mode_even_across_wraps():
    time = t([0., 5., 100., 600.], requires_grad=True)
    omega = .02+3e-5*time
    phi = 2.9+.02*time+1.5e-5*time*time
    energy = torch.sqrt(1-.21*omega.pow(2/3))
    d = delta22(omega, **phase_inputs(hreal_over_M=energy))
    phase = phase22(omega, phi, hreal_over_M=energy, residual_phase=d)
    h = mode22(omega, phi, residual_phase=d, **mode_inputs(
        omega_zero_radial=omega, hreal_over_M=energy, heff_over_mu=1-omega.pow(2/3)/2))
    frequency, = torch.autograd.grad(phase.sum(), (time,), retain_graph=True)
    hdot_r, = torch.autograd.grad(h.real.sum(), (time,), retain_graph=True)
    hdot_i, = torch.autograd.grad(h.imag.sum(), (time,))
    torch.testing.assert_close(frequency, -((hdot_r+1j*hdot_i)/h).imag, atol=3e-17, rtol=3e-15)
    assert bool(torch.all(frequency > 0))
    assert bool(torch.all((frequency-2*omega).abs() > 1e-5))
    assert phase[-1]-phase[0] > 10*math.pi


def test_mode_modulus_orbital_rotation_batch_and_body_exchange():
    omega = t([[.01, .02, .06]])
    x1 = t([[.5], [.8]])
    chi1, chi2 = t([[.2], [-.6]]), t([[-.1], [.3]])
    p = phase_inputs(x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)
    m = mode_inputs(x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)
    d = delta22(omega, **p)
    h = mode22(omega, t([[.3], [.8]]), residual_phase=d, **m)
    assert h.shape == (2, 3)
    torch.testing.assert_close(h.abs(), seobnrv4_published_amplitude22(omega, **m))
    shifted = mode22(omega, t([[.3], [.8]])+.19, residual_phase=d, **m)
    torch.testing.assert_close(shifted, h*np.exp(-.38j))
    p.update(x1=1-x1, x2=x1, chi1=chi2, chi2=chi1)
    m.update(x1=1-x1, x2=x1, chi1=chi2, chi2=chi1)
    swapped = mode22(omega, t([[.3], [.8]]), residual_phase=delta22(omega, **p), **m)
    torch.testing.assert_close(swapped, h)
    for i in range(2):
        row_p = {k: v[i, 0] if isinstance(v, torch.Tensor) and v.shape == (2, 1) else v for k, v in p.items()}
        row_m = {k: v[i, 0] if isinstance(v, torch.Tensor) and v.shape == (2, 1) else v for k, v in m.items()}
        scalar = mode22(omega[0], [.3, .8][i], residual_phase=delta22(omega[0], **row_p), **row_m)
        torch.testing.assert_close(scalar, h[i])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inference_dtype_and_zero_mass_endpoint(dtype):
    omega = torch.tensor([.01, .05], dtype=dtype)
    with torch.inference_mode():
        d = delta22(omega, **phase_inputs(x1=1., x2=0.))
        h = mode22(omega, .3, residual_phase=d, **mode_inputs(x1=1., x2=0.))
        nonzero = mode22(omega, .3, residual_phase=d, **mode_inputs())
    assert h.dtype == (torch.complex64 if dtype == torch.float32 else torch.complex128)
    assert bool(torch.all(h == 0))
    assert bool(torch.all(torch.isfinite(nonzero)))


@pytest.mark.parametrize("x1", [.5, .8])
def test_end_to_end_complex_gradcheck(x1):
    args = [t(value, requires_grad=True) for value in (.028, .033, .91, .98, x1, .6, -.2, .43)]

    def function(omega, omega0, heff, energy, a, c1, c2, phi):
        d = delta22(omega, **phase_inputs(hreal_over_M=energy, x1=a, x2=1-a,
                                        chi1=c1, chi2=c2, include_positive_chi_540=True))
        return mode22(omega, phi, residual_phase=d, **mode_inputs(
            omega_zero_radial=omega0, heff_over_mu=heff, hreal_over_M=energy,
            x1=a, x2=1-a, chi1=c1, chi2=c2))

    assert torch.autograd.gradcheck(function, tuple(args), eps=1e-6, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("change", [dict(spin_v6="default"), dict(include_pan_v8=1),
    dict(include_positive_chi_540=None), dict(x1=.8, x2=.3), dict(chi1=1.1),
    dict(hreal_over_M=-.1), dict(hreal_over_M=float("nan"))])
def test_invalid_phase_inputs(change):
    with pytest.raises(ValueError):
        delta22(t(.02), **phase_inputs(**change))


def test_invalid_phase_and_frequency_do_not_silently_cast_complex_inputs():
    with pytest.raises(ValueError):
        phase22(t(.02), .4, hreal_over_M=1., residual_phase=1j)
    with pytest.raises(ValueError):
        tail22(torch.tensor([1], dtype=torch.int64), hreal_over_M=1.)
    with pytest.raises(ValueError):
        tail22(t(0.), hreal_over_M=1.)
