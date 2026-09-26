"""Paper-derived tidal checks and normalized numerical regression tests."""
import math

import pytest
import torch

from torchwave.constants import PI, MSUN_SEC, MPC_SI, C_SI
from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal, IMRPhenomD_NRTidalv2,
    _bns_merger_frequency, _nrtidal_amplitude, _planck_taper_window,
)


def test_merger_frequency_published_effective_coupling():
    # Eq. 11 and the preceding coupling substitution, arXiv:1804.02235.
    m1, m2, l1, l2 = 1.6, 1.1, 200., 800.
    xa, xb = m1/(m1+m2), m2/(m1+m2)
    kappa = 3/13*((xa+12*xb)*xa**4*l1+(xb+12*xa)*xb**4*l2)
    kappa2 = 3*(xa**4*xb*l1+xb**4*xa*l2)
    def frequency(k):
        return .3586*math.sqrt(xb/xa)*(1+.03354*k+.00004315*k*k)/(1+.07542*k+.0002236*k*k)/(2*PI*(m1+m2)*MSUN_SEC)
    values = [torch.tensor(v, dtype=torch.float64) for v in (m1,m2,l1,l2)]
    expected = torch.tensor(frequency(kappa), dtype=torch.float64)
    torch.testing.assert_close(_bns_merger_frequency(*values), expected, rtol=1e-14, atol=0)
    torch.testing.assert_close(_bns_merger_frequency(values[1],values[0],values[3],values[2]), expected, rtol=1e-14, atol=0)
    assert abs(frequency(kappa2)/frequency(kappa)-1) > 1e-5


def test_additive_amplitude_low_frequency_pn():
    # Independent 6PN Eq. 22 / Section 1.4 of parity-calibration-spec.md
    x = torch.tensor([1e-5, 2e-5], dtype=torch.float64)
    m_sec, d_sec, lam = 2.8 * MSUN_SEC, 100 * MPC_SI / C_SI, 400.0
    kappa = (3.0 / 16.0) * lam
    prefactor = math.sqrt(5.0 * math.pi * 0.25 / 24.0) * m_sec**2 / d_sec
    expected = prefactor * (-9.0 * kappa * x**(13.0 / 4.0) * (1.0 + (449.0 / 108.0) * x))
    actual = _nrtidal_amplitude(x, m_sec, d_sec, 0.25, lam)
    torch.testing.assert_close(actual / expected, torch.ones_like(x), rtol=1e-7, atol=0)
    # Recover the independently specified NLO coefficient, not only the leading power.
    leading = prefactor * (-9.0 * kappa * x**(13.0 / 4.0))
    torch.testing.assert_close((actual / leading - 1) / x, torch.full_like(x, 449.0 / 108.0), rtol=1e-4, atol=0)


@pytest.mark.parametrize('inclination', [0., .7, math.pi/2, 2.4])
def test_additive_amplitude_polarization_projection(inclination):
    f = torch.tensor([40., 200., 700., 1200.], dtype=torch.float64)
    kw = dict(f=f, mass1=1.4, mass2=1.3, distance=80., inclination=inclination)
    hp0, hc0 = IMRPhenomD()(**kw)
    hp, hc = IMRPhenomD_NRTidalv2()(**kw, lambda1=350., lambda2=420., apply_merger_taper=False)
    xa, xb = 1.4 / 2.7, 1.3 / 2.7
    kappa = 3 / 13 * ((xa + 12 * xb) * xa**4 * 350 + (xb + 12 * xa) * xb**4 * 420)
    x = (PI * 2.7 * MSUN_SEC * f) ** (2 / 3)
    # Dietrich et al. (2019), arXiv:1905.06011 Eq. 24 prints d=13477.8.
    c_amp = 13477.8
    eta = 1.4 * 1.3 / 2.7**2
    at = math.sqrt(5 * math.pi * eta / 24) * (2.7 * MSUN_SEC)**2 / (80 * MPC_SI / C_SI) * (-9 * kappa * x**(13/4) * (1 + 449/108 * x + 22672/9 * x**2.89) / (1 + c_amp * x**4))
    plus = .5 * (1 + math.cos(inclination)**2)
    torch.testing.assert_close((hp.abs() - hp0.abs()) / 1e-23, plus * at / 1e-23, rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(hc / 1e-23, hp / 1e-23 * (-1j * math.cos(inclination) / plus), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('version', ['v1','v2'])
def test_zero_tidal_recovers_bbh(version):
    f = torch.tensor([30.,100.,500.,1200.], dtype=torch.float64)
    kw = dict(f=f,mass1=1.4,mass2=1.3,chi1=.05,chi2=-.03,distance=50.,inclination=.4)
    expected = IMRPhenomD()(**kw)
    actual = IMRPhenomD_NRTidal(version)(**kw,lambda1=0.,lambda2=0.,apply_merger_taper=False)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a/1e-23,b/1e-23,rtol=1e-13,atol=0)


def test_dtype_contracts():
    f=torch.linspace(30.,800.,30,dtype=torch.float32)
    kw=dict(mass1=1.4,mass2=1.3,lambda1=350.,lambda2=420.)
    direct=IMRPhenomD_NRTidalv2()(f=f,**kw)
    public=get_fd_waveform(approximant='IMRPhenomD_NRTidalv2',sample_frequencies=f,**kw)
    for a,b in zip(direct,public):
        assert a.dtype == torch.complex64 and b.dtype == torch.complex128
        assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_planck_taper_monotonicity_and_bounds():
    f=torch.linspace(500.,1500.,2001,dtype=torch.float64)
    w=_planck_taper_window(f,torch.tensor(1000.),torch.tensor(1200.))
    assert (w[f<=1000]==1).all() and (w[f>=1200]==0).all()
    assert ((w>=0)&(w<=1)).all() and (torch.diff(w)<=0).all()


def test_complex_waveform_gradcheck_with_taper_and_zero_bins():
    vals = torch.tensor([1.45,1.35,.015,-.012,350.,400.],dtype=torch.float64)
    fm = _bns_merger_frequency(vals[0],vals[1],vals[4],vals[5]).item()
    f=torch.tensor([-1.,0.,40.,200.,.9*fm,1.1*fm,1.3*fm,1e5],dtype=torch.float64)
    model=IMRPhenomD_NRTidalv2()
    p=torch.tensor([1.45,1.35,.015,-.012,.875,1.,1.,.4],dtype=torch.float64,requires_grad=True)
    def fn(p):
        hp,hc=model(f=f,mass1=p[0],mass2=p[1],chi1=p[2],chi2=p[3],lambda1=400*p[4],lambda2=400*p[5],distance=100*p[6],inclination=p[7])
        return torch.view_as_real(torch.stack((hp,hc))/1e-23)
    actual=fn(p)
    assert torch.isfinite(actual).all()
    assert (actual[:,[0,1,6,7]]==0).all()
    assert torch.autograd.gradcheck(fn,(p,),eps=1e-7,atol=3e-5,rtol=3e-4)


def test_batched_vs_scalar_with_distance_and_inclination():
    f=torch.tensor([30.,100.,700.,1700.,2200.],dtype=torch.float64)
    params=torch.tensor([[1.45,1.35,.03,-.01,320,410,70,0],
                         [1.25,1.55,.02,-.02,360,250,100,.3],
                         [1.38,1.28,0,.015,480,520,120,1.57]],dtype=torch.float64)
    keys=('mass1','mass2','chi1','chi2','lambda1','lambda2','distance','inclination')
    model=IMRPhenomD_NRTidalv2()
    batch=model(f=f,**dict(zip(keys,params.T)))
    for i,row in enumerate(params):
        scalar=model(f=f,**dict(zip(keys,row)))
        for a,b in zip(batch,scalar):
            assert torch.linalg.vector_norm(a[i]-b)/torch.linalg.vector_norm(b) < 1e-12
