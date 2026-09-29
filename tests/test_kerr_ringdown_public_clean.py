import math
import torch
import pytest
from torchwave.approximants.kerr_ringdown_public_clean import kerr220_td, kerr220_fd
from torchwave.waveforms import get_fd_waveform

def test_td_fd_shapes_and_broadcast():
    t=torch.linspace(0, .2, 8,dtype=torch.float64)
    hp,hc=kerr220_td(t,65.,.7,100.,inclination=.8,phic=.2)
    assert hp.shape==t.shape and torch.isfinite(hp).all() and torch.isfinite(hc).all()
    fp,fc=kerr220_fd(torch.tensor([0.,10.]),65.,.7,100.,inclination=.8,phic=.2)
    assert fp.shape==(2,) and torch.isfinite(fp.real).all()

def test_scaling_and_phase_and_tc():
    t=torch.tensor([.01,.02],dtype=torch.float64)
    a=kerr220_td(t,65.,.7,100.,energy_fraction=.01,inclination=.8,phic=0.)
    b=kerr220_td(t,65.,.7,200.,energy_fraction=.01,inclination=.8,phic=0.)
    assert torch.allclose(a[0]/b[0],torch.tensor(2.,dtype=torch.float64),rtol=1e-10)
    z=kerr220_td(torch.tensor([-.01,.01]),65.,.7,100.,tc=.01)
    assert z[0][0]==0

def test_autograd_and_float32():
    s=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    y=kerr220_td(torch.tensor(.02,dtype=torch.float64),65.,s,100.,inclination=.8)[0]
    assert torch.isfinite(torch.autograd.grad(y,s)[0])
    q=kerr220_fd(torch.tensor([10.],dtype=torch.float32),65.,torch.tensor(.3),100.)
    assert q[0].dtype==torch.complex64

def test_parameter_batch_grid_and_safe_preonset():
    t=torch.linspace(-1e6, 1e-2, 8, dtype=torch.float32)
    hp,hc=kerr220_td(t, torch.tensor([65.,70.]), torch.tensor([.3,.7]), 100., tc=1e5)
    assert hp.shape==(2,8) and torch.isfinite(hp).all() and (hp[:,0]==0).all()
    hp,_=kerr220_td(t, torch.ones((2,3))*65, torch.ones((2,3))*.7, 100., tc=1e5)
    assert hp.shape==(2,3,8)

def test_fd_time_shift_law():
    f=torch.tensor([10.,20.],dtype=torch.float64)
    a=kerr220_fd(f,65.,.7,100.,inclination=.8,phic=.2,tc=0.)[0]
    b=kerr220_fd(f,65.,.7,100.,inclination=.8,phic=.2,tc=.01)[0]
    assert torch.allclose(b, a*torch.exp(-2j*math.pi*f*.01), rtol=2e-12, atol=0)

def test_public_dispatch_routes_phic_tc_and_rejects_implicit_modes():
    f=torch.tensor([10.,20.],dtype=torch.float64)
    direct=kerr220_fd(f,65.,.7,100.,inclination=.8,phic=.2,tc=.01)
    routed=get_fd_waveform("KerrRingdown",65.,1.,spin1z=.7,distance=100.,
                           inclination=.8,phic=.2,tc=.01,
                           sample_frequencies=f)
    assert torch.allclose(routed[0],direct[0],atol=0,rtol=0)
    with pytest.raises(ValueError, match="explicit amplitudes"):
        get_fd_waveform("ringdown",65.,1.,spin1z=.7,
                        sample_frequencies=f,modes=[(2,2)])

def test_waveform_finite_difference_mass_spin_and_inclination():
    f=torch.tensor([30.],dtype=torch.float64)
    def q(m,a,i):
        return kerr220_fd(f,m,a,100.,inclination=i,phic=.37)[0].real
    m,a,i=65.,.7,.8; h=1e-5
    dm=(q(m+h,a,i)-q(m-h,a,i))/(2*h)
    da=(q(m,a+h,i)-q(m,a-h,i))/(2*h)
    di=(q(m,a,i+h)-q(m,a,i-h))/(2*h)
    assert torch.isfinite(torch.cat((dm,da,di))).all()
    assert dm.abs().item()>0 and da.abs().item()>0 and di.abs().item()>0
    mt=torch.tensor(m,dtype=torch.float64,requires_grad=True)
    at=torch.tensor(a,dtype=torch.float64,requires_grad=True)
    y=kerr220_fd(f,mt,at,100.,inclination=i,phic=.37)[0].real
    gm,ga=torch.autograd.grad(y,(mt,at))
    assert torch.allclose(gm,dm,rtol=3e-5,atol=0)
    assert torch.allclose(ga,da,rtol=3e-5,atol=0)
    it=torch.tensor(i,dtype=torch.float64,requires_grad=True)
    pt=torch.tensor(.37,dtype=torch.float64,requires_grad=True)
    yp=kerr220_fd(f,m,a,100.,inclination=it,phic=pt)[0].real
    gi,gp=torch.autograd.grad(yp,(it,pt))
    di=(q(m,a,i+h)-q(m,a,i-h))/(2*h)
    dp=(kerr220_fd(f,m,a,100.,inclination=i,phic=.37+h)[0].real-
        kerr220_fd(f,m,a,100.,inclination=i,phic=.37-h)[0].real)/(2*h)
    assert torch.allclose(gi,di,rtol=3e-5,atol=0)
    assert torch.allclose(gp,dp,rtol=3e-5,atol=0)

def test_optional_public_lal_fd_reference():
    """Compare the clean formula with the public TD callable when LAL is present."""
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    import numpy as np
    from kerr_public_oracle import public_ringdown_fd
    from kerr_public_oracle import public_ringdown_fd_converged
    for spin, inc, phase in ((.7,0.,0.),(.3,.8,.2),(.95,.8,.2),(.9,.8,.2)):
        f, hp_ref, hc_ref, _ = public_ringdown_fd_converged(
            65., spin, 100., inclination=inc, phase=phase,
            sample_rate=32768, max_sample_rate=131072, delta_f=.25,
            comparison_band=(50.,500.), f_max=500.)
        hp, hc = kerr220_fd(torch.from_numpy(f),65.,spin,100.,
                            inclination=inc,phic=phase)
        mask=(f>=50.)&(f<=500.)
        def rel(x,y): return np.linalg.norm(x[mask]-y[mask])/np.linalg.norm(y[mask])
        assert rel(hp.detach().numpy(),hp_ref) < 1e-6
        assert rel(hc.detach().numpy(),hc_ref) < 1e-6


def test_fixed_off_axis_energy_onset_holdout():
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    from tests.parity_harness import get_canonical_holdout_cases, run_oracle_comparison
    case = next(c for c in get_canonical_holdout_cases()
                if c["id"] == "KerrRingdown_holdout_off_axis_energy_onset")
    result = run_oracle_comparison(case, strict=True)
    assert result["status"] == "PASSED", result
