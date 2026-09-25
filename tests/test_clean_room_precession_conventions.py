"""Published frame and Table IV identities; no external waveform oracle."""
import math

import pytest
import torch

from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.precession_msa import (
    compute_l_3pn, evaluate_msa_dynamics, evaluate_msa_precession,
)
from torchwave.approximants.precession_beta_clean import cos_beta
from torchwave.constants import MSUN_SEC

DTYPE = torch.float64


def t(x):
    return torch.tensor(x, dtype=DTYPE)


def parameters():
    return dict(v=t([.18,.24,.30]),v0=t(.24),m1=t(40.),m2=t(20.),
                s1x=t(.3),s1y=t(.2),s1z=t(.4),
                s2x=t(-.2),s2y=t(.1),s2z=t(-.2),
                iota=t(.7),phi_ref=t(.3))


def reference_geometry(p):
    """Direct Cartesian basis construction, independent of Euler formulas."""
    eta = p['m1']*p['m2']/(p['m1']+p['m2'])**2
    n1, n2 = p['m1']/(p['m1']+p['m2']), p['m2']/(p['m1']+p['m2'])
    spin = torch.stack([n1*n1*p['s1'+a]+n2*n2*p['s2'+a] for a in 'xyz'])
    l0 = compute_l_3pn(p['v0'],eta,n1-n2,spin[2],n2*p['s2z']-n1*p['s1z'])
    j = spin+torch.stack((t(0.),t(0.),l0))
    jhat = j/torch.linalg.vector_norm(j)
    inc, phi = p['iota'], p['phi_ref']
    n = torch.stack((torch.sin(inc)*torch.sin(phi),torch.sin(inc)*torch.cos(phi),torch.cos(inc)))
    observer_x = torch.stack((-torch.cos(inc)*torch.sin(phi),-torch.cos(inc)*torch.cos(phi),torch.sin(inc)))
    yj = torch.linalg.cross(jhat,n)
    yj = yj/torch.linalg.vector_norm(yj)
    xj = torch.linalg.cross(yj,jhat)
    alpha = torch.atan2(yj[2],xj[2])
    phi_j = torch.atan2(j[1],j[0])
    theta = torch.acos(torch.dot(jhat,n))
    x_sphere = torch.cos(theta)*xj-torch.sin(theta)*jhat
    zeta = torch.atan2(torch.dot(observer_x,yj),torch.dot(observer_x,x_sphere))
    return alpha,phi_j-math.pi,theta,zeta,phi_j


def raw_angles(p,version):
    n1, n2 = p['m1']/(p['m1']+p['m2']), p['m2']/(p['m1']+p['m2'])
    eta, delta = n1*n2,n1-n2
    if version == 223:
        s1 = n1*n1*torch.stack([p['s1'+a] for a in 'xyz'])
        s2 = n2*n2*torch.stack([p['s2'+a] for a in 'xyz'])
        l = compute_l_3pn(p['v'],eta,delta,s1[2]+s2[2],n2*p['s2z']-n1*p['s1z'])
        alpha,_,gamma,valid = evaluate_msa_dynamics(
            p['v'],p['v0'],eta,delta,n2/n1,s1,s2,l,anchor_reference=False,
        )
        assert valid
        return alpha,-gamma
    a1,a2 = 2+1.5*n2/n1,2+1.5*n1/n2
    chip = torch.maximum(a1*n1*n1*torch.hypot(p['s1x'],p['s1y']),
                         a2*n2*n2*torch.hypot(p['s2x'],p['s2y']))/(a1*n1*n1)
    coeff = IMRPhenomP.nnlo_angle_coefficients(eta,delta,n1,chip,n1*p['s1z']+n2*p['s2z'])
    v = p['v']
    basis = torch.stack((v**-3,v**-2,v**-1,v,torch.log(v**3)))
    return (torch.stack(coeff[:5])[:,None]*basis).sum(0),(torch.stack(coeff[5:])[:,None]*basis).sum(0)


@pytest.mark.parametrize('version',[102,223])
@pytest.mark.parametrize('convention',[0,1,5,6,7])
def test_table_iv_angles_and_carrier_phase(convention,version):
    p = parameters()
    alpha0,eps0,theta,zeta,phi_j = reference_geometry(p)
    ar,er = raw_angles(p,version)
    expected_a = ar+alpha0
    expected_e = er
    if convention in (0,1,6):
        expected_a = expected_a-ar[1]
        expected_e = expected_e-er[1]
    if convention in (1,6):
        expected_e = expected_e+eps0
    actual = evaluate_msa_precession(**p,prec_version=version,convention=convention,return_carrier_phase=True)
    # Euler branches differing by 2pi describe the same rotation.
    torch.testing.assert_close(torch.exp(1j*actual[0]),torch.exp(1j*expected_a),atol=3e-13,rtol=3e-13)
    torch.testing.assert_close(actual[2],-expected_e,atol=3e-13,rtol=3e-13)
    torch.testing.assert_close(actual[3],theta,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(torch.exp(1j*actual[4]),torch.exp(1j*zeta),atol=2e-14,rtol=2e-14)
    expected_phase = phi_j if convention == 0 else t(0.) if convention == 1 else p['phi_ref']
    torch.testing.assert_close(actual[5],expected_phase)
    assert len(evaluate_msa_precession(**p,prec_version=version,convention=convention)) == 5


def test_msa_evaluator_explicit_none_matches_declared_default():
    p = parameters()
    default = evaluate_msa_precession(**p, return_carrier_phase=True)
    explicit_none = evaluate_msa_precession(**p, prec_version=None, return_carrier_phase=True)
    for expected, actual in zip(default, explicit_none):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('version',[102,223])
def test_unanchored_phases_retain_reference_value_and_same_evolution(version):
    p = parameters()
    raw_a,raw_e = raw_angles(p,version)
    assert raw_a[1].abs() > 1 and raw_e[1].abs() > 1
    anchored = evaluate_msa_precession(**p,prec_version=version,convention=6)
    raw = evaluate_msa_precession(**p,prec_version=version,convention=7)
    torch.testing.assert_close(raw[0]-raw[0][1],anchored[0]-anchored[0][1],atol=3e-13,rtol=3e-13)
    torch.testing.assert_close(raw[2]-raw[2][1],anchored[2]-anchored[2][1],atol=3e-13,rtol=3e-13)


@pytest.mark.parametrize('mode',[None,(2,1),(2,2),(3,2),(3,3),(4,4)])
def test_wrappers_apply_table_iv_phase_to_each_mode(mode):
    model = IMRPhenomXP() if mode is None else IMRPhenomXPHM(modes=[mode])
    order = 2 if mode is None else mode[1]
    f = t([25.,40.,80.])
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=.3,f_ref=40.)
    default = model(f,**p,convention=1)
    phi_j = torch.atan2(t(4*.2+.1),t(4*.3-.2))
    # XP's convention-0 and convention-1 full strains differ by a constant
    # sign at this generic point.  Appendix C Eq. (79) rotates alpha and
    # gamma with the Table-IV carrier, and Eq. (81) fixes the output basis.
    # The aligned-boundary suite also checks both strains against public XP.
    phase_zero = t(math.pi) if mode is None else order*(2*phi_j-math.pi)
    for convention, phase in [(0,phase_zero),(6,t(order*.3))]:
        for h,h0 in zip(model(f,**p,convention=convention),default):
            torch.testing.assert_close(h,torch.exp(1j*phase)*h0,atol=1e-34,rtol=2e-11)
    for h5,h7 in zip(model(f,**p,convention=5),model(f,**p,convention=7)):
        torch.testing.assert_close(h5,h7,atol=1e-34,rtol=2e-11)


def test_xp_v223_convention6_batched_phase_and_gradient():
    model = IMRPhenomXP()
    f = t([25.,40.,80.])
    phi = t([-.4,.3]).requires_grad_()
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,f_ref=40.,
             convention=6)
    batch = model(f,phic=phi,**p)
    for row in range(2):
        scalar = model(f,phic=phi[row],**p)
        for actual,expected in zip(batch,scalar):
            torch.testing.assert_close(actual[row],expected,atol=1e-34,rtol=2e-11)

    def loss(angle):
        hp,hc = model(f,phic=angle,**p)
        return 1e22*(hp.real+.3*hc.imag).sum()

    gradient = torch.autograd.grad(loss(phi),phi)[0]
    for row in range(2):
        shift = torch.zeros_like(phi)
        shift[row] = 1e-5
        finite_diff = (loss(phi.detach()+shift)-loss(phi.detach()-shift))/2e-5
        torch.testing.assert_close(gradient[row],finite_diff,atol=2e-8,rtol=2e-7)


@pytest.mark.parametrize("f_ref,spin1z",((40.,-.95),(25.,-.18)))
@pytest.mark.parametrize("spin1x",(0.,1e-9))
def test_xp_v223_convention6_keeps_opposed_j_guard(f_ref,spin1z,spin1x):
    # For the second row L+S_parallel is positive at f_ref=25 Hz but
    # negative at an evaluated frequency (40 Hz). The first is already
    # opposed at f_ref. Convention 6 has no aligned-carrier exemption.
    f = t([25.,40.,80.])
    with pytest.raises(ValueError,match='L \\+ S_parallel <= 0'):
        IMRPhenomXP()(f,mass1=100.,mass2=5.,spin1x=spin1x,spin1z=spin1z,
                      inclination=.7,phic=.3,f_ref=f_ref,convention=6)


def test_xp_v223_convention6_mixed_opposed_j_batch_keeps_legacy_exception():
    f = t([25.,40.,80.])
    p = dict(mass1=t([40.,100.]),mass2=t([20.,5.]),spin1x=t([.3,0.]),
             spin1z=t([.4,-.95]),spin2z=-.2,inclination=.7,phic=.3,
             f_ref=40.,convention=6)
    model = IMRPhenomXP()
    with pytest.raises(ValueError,match='L \\+ S_parallel <= 0') as current:
        model(f,**p)
    with pytest.raises(ValueError,match='L \\+ S_parallel <= 0') as legacy:
        model(f,**p,_force_legacy=True)
    assert str(current.value) == str(legacy.value)


@pytest.mark.parametrize("options",(
    {"phi0": .4},
    {"prec_version": 102},
    {"final_spin_mod": 0},
))
def test_xp_convention6_other_tuples_keep_legacy_route(options):
    model = IMRPhenomXP()
    f = t([25.,40.,80.])
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=.3,
             f_ref=40.,convention=6,**options)
    current = model(f,**p)
    legacy = model(f,**p,_force_legacy=True)
    for actual,expected in zip(current,legacy):
        torch.testing.assert_close(actual,expected,atol=1e-34,rtol=2e-11)


def test_xp_convention6_mixed_azimuth_rows_match_scalar_routes():
    model = IMRPhenomXP()
    f = t([25., 40., 80.])
    phi0 = t([0., .4])
    p = dict(mass1=40., mass2=20., spin1x=.3, spin1y=.2, spin1z=.4,
             spin2x=-.2, spin2y=.1, spin2z=-.2, inclination=.7, phic=.3,
             f_ref=40., convention=6)
    batch = model(f, **p, phi0=phi0)
    for row in range(2):
        scalar = model(f, **p, phi0=phi0[row])
        for actual, expected in zip(batch, scalar):
            torch.testing.assert_close(actual[row], expected, atol=0., rtol=0.)


@pytest.mark.parametrize('convention',[0,1,5,6,7])
def test_batched_source_azimuth_and_gradient_match_scalar_calls(convention):
    model = IMRPhenomXPHM(modes=[(2,2),(3,3)])
    f = t([25.,40.,80.])
    phi = t([-.4,.3]).requires_grad_()
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,f_ref=40.,convention=convention)
    batched = model(f,phic=phi,**p)
    for row in range(2):
        for hb,hs in zip(batched,model(f,phic=phi[row],**p)):
            torch.testing.assert_close(hb[row],hs,atol=1e-34,rtol=2e-11)
    def loss(angle):
        hp,hc = model(f,phic=angle,**p)
        return 1e22*(hp.real+.3*hc.imag).sum()
    grad = torch.autograd.grad(loss(phi),phi)[0]
    step = 1e-5
    for row in range(2):
        displacement = torch.zeros_like(phi)
        displacement[row] = step
        finite_diff = (loss(phi.detach()+displacement)-loss(phi.detach()-displacement))/(2*step)
        torch.testing.assert_close(grad[row],finite_diff,atol=2e-8,rtol=2e-7)
    assert torch.isfinite(grad).all() and (grad.abs() > 1e-5).all()


@pytest.mark.parametrize('mode',[None,(2,1),(3,3)])
def test_aligned_table_limits_and_endpoint_contracts(mode):
    model = IMRPhenomXP() if mode is None else IMRPhenomXPHM(modes=[mode])
    f = t([25.,40.,80.])
    p = dict(mass1=40.,mass2=20.,spin1z=.4,spin2z=-.2,inclination=.7,phic=.3,f_ref=40.)
    default = model(f,**p)
    # Public XP now uses the physical zero-nutation opening here. Its explicit
    # v223 compatibility helper retains the finite jump and is tested
    # separately. XPHM retains its existing strict continuity contract.
    for azimuth in [-2.,.4,2.1]:
        near = model(f,**p,spin1x=1e-9*math.cos(azimuth),spin1y=1e-9*math.sin(azimuth))
        for h,h0 in zip(near,default):
            relative = torch.linalg.vector_norm(h-h0)/torch.linalg.vector_norm(h0)
            assert relative < 2e-8
    for convention in [0,5,6,7]:
        exact = model(f,**p,convention=convention)
        # The declared zero-spin azimuth is phi_JL=0. Approach that gauge.
        near = model(f,**p,spin1x=1e-9,convention=convention)
        for h,h0 in zip(near,exact):
            assert torch.isfinite(h0).all()
            relative = torch.linalg.vector_norm(h-h0)/torch.linalg.vector_norm(h0)
            assert relative < 2e-8
    order = 2 if mode is None else mode[1]
    for h,h0 in zip(model(f,**p,convention=6),default):
        torch.testing.assert_close(h,torch.exp(1j*t(order*.3))*h0,atol=1e-34,rtol=2e-11)


def test_paper_beta_has_strict_aligned_limit():
    """Pratten et al. Eq. (17): no transverse spin gives cos(beta)=1."""
    f = t([25.,40.,80.])
    aligned = cos_beta(40.,20.,f,0.,0.,.4,0.,0.,-.2)
    torch.testing.assert_close(aligned,torch.ones_like(aligned),atol=0.,rtol=0.)
    near = cos_beta(40.,20.,f,1e-3,0.,.4,0.,0.,-.2)
    assert torch.all(near < aligned)
    assert torch.all((aligned-near) < 1e-6)


@pytest.mark.parametrize('model',[IMRPhenomXP,IMRPhenomXPHM])
def test_invalid_convention_is_rejected_in_aligned_fast_path(model):
    with pytest.raises(ValueError,match='convention'):
        model()(t([25.,40.]),convention=2)


def test_pv2_cartesian_alpha_covers_quadrants_and_small_cones():
    for spin_azimuth in [-2.7,-.2,1.3,2.8]:
        for magnitude in [.3,1e-12]:
            p = parameters()
            p.update(s1x=t(magnitude*math.cos(spin_azimuth)),
                     s1y=t(magnitude*math.sin(spin_azimuth)),s2x=t(0.),s2y=t(0.))
            p['v'] = p['v0'][None]
            expected,_,_,_,_ = reference_geometry(p)
            result = evaluate_msa_precession(**p,prec_version=102,convention=1)
            torch.testing.assert_close(torch.exp(1j*result[0][0]),torch.exp(1j*expected),atol=3e-13,rtol=3e-13)


@pytest.mark.parametrize('model',[IMRPhenomXP,IMRPhenomXPHM])
@pytest.mark.parametrize('version',[102,223])
def test_reference_frequency_alone_broadcasts_and_differentiates(model,version):
    waveform = model()
    f = t([25.,40.,80.])
    fref = t([30.,40.]).requires_grad_()
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=.3,prec_version=version)
    batched = waveform(f,**p,f_ref=fref)
    for row in range(2):
        for hb,hs in zip(batched,waveform(f,**p,f_ref=fref[row])):
            torch.testing.assert_close(hb[row],hs,atol=1e-34,rtol=2e-11)
    loss = 1e22*(batched[0].real+.3*batched[1].imag).sum()
    grad = torch.autograd.grad(loss,fref)[0]
    assert torch.isfinite(grad).all()
    step = 1e-4
    for row in range(2):
        h_up = waveform(f,**p,f_ref=fref[row].detach()+step)
        h_dn = waveform(f,**p,f_ref=fref[row].detach()-step)
        fd = 1e22*((h_up[0]-h_dn[0]).real+.3*(h_up[1]-h_dn[1]).imag).sum()/(2*step)
        torch.testing.assert_close(grad[row],fd,atol=2e-7,rtol=2e-6)


@pytest.mark.parametrize('mode',[None,(2,1),(2,2),(3,2),(3,3),(4,4)])
def test_nonzero_observer_azimuth_has_physical_aligned_limit(mode):
    waveform = IMRPhenomXP() if mode is None else IMRPhenomXPHM(modes=[mode])
    f = t([25.,40.,80.])
    p = dict(mass1=40.,mass2=20.,spin1z=.4,spin2z=-.2,inclination=.7,phic=.3,f_ref=40.)
    order = 2 if mode is None else mode[1]
    phi0 = t(.4).requires_grad_()
    zero_azimuth = waveform(f,**p)
    exact = waveform(f,**p,phi0=phi0)
    for h,h0 in zip(exact,zero_azimuth):
        torch.testing.assert_close(h,torch.exp(-1j*order*phi0)*h0,atol=1e-34,rtol=2e-11)
    derivative = torch.autograd.grad(1e22*exact[0].real.sum(),phi0)[0]
    torch.testing.assert_close(derivative,1e22*(-1j*order*exact[0]).real.sum(),atol=2e-12,rtol=2e-12)
    for convention in (1,6):
        exact = waveform(f,**p,phi0=phi0,convention=convention)
        for azimuth in (-2.,.4,2.1):
            near = waveform(f,**p,phi0=phi0,convention=convention,
                            spin1x=1e-9*math.cos(azimuth),spin1y=1e-9*math.sin(azimuth))
            for h,h0 in zip(near,exact):
                assert torch.linalg.vector_norm(h-h0)/torch.linalg.vector_norm(h0) < 2e-8


@pytest.mark.parametrize('mode', IMRPhenomXHM.SUPPORTED_MODES)
def test_xhm_positive_frequency_observer_azimuth(mode):
    # Appendix B, Eqs. B6-B7: positive frequency carries the -m harmonic.
    # Its azimuthal factor is exp(-i*m*phi0) for each aligned carrier mode.
    model = IMRPhenomXHM(modes=[mode])
    f = t([25., 40., 80.])
    p = dict(mass1=40., mass2=20., chi1=.4, chi2=-.2,
             inclination=.7, phic=.3, f_ref=40.)
    phi0 = t(.4).requires_grad_()
    reference = model(f, **p)
    rotated = model(f, **p, phi0=phi0)
    for h, h0 in zip(rotated, reference):
        torch.testing.assert_close(h, torch.exp(-1j*mode[1]*phi0)*h0,
                                   atol=1e-34, rtol=2e-13)
    derivative = torch.autograd.grad(1e22*rotated[0].real.sum(), phi0)[0]
    torch.testing.assert_close(derivative,
                               1e22*(-1j*mode[1]*rotated[0]).real.sum(),
                               atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('convention',[0,1,5,6,7])
@pytest.mark.parametrize('method,version,remnant',[
    ('nnlo',102,0),('msa',223,3),
])
def test_xp_xphm22_agree_for_batched_observer_azimuth(convention,method,version,remnant):
    f = t([25.,40.,80.])
    phi0 = t([-.4,.3]).requires_grad_()
    p = dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,
             spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=.3,f_ref=40.,
             convention=convention,precession_method=method,prec_version=version,
             final_spin_mod=remnant)
    # The published 22-only identity compares the same angle/remnant model.
    # Both public defaults are MSA v223/remnant 3; the explicit tuples also
    # protect the nondefault NNLO reduction.
    for hp,hpm in zip(IMRPhenomXP()(f,**p,phi0=phi0),IMRPhenomXPHM(modes=[(2,2)])(f,**p,phi0=phi0)):
        torch.testing.assert_close(hp,hpm,atol=1e-34,rtol=2e-11)
        gp = torch.autograd.grad(1e22*hp.real.sum(),phi0,retain_graph=True)[0]
        gpm = torch.autograd.grad(1e22*hpm.real.sum(),phi0,retain_graph=True)[0]
        torch.testing.assert_close(gp,gpm,atol=2e-11,rtol=2e-11)


@pytest.mark.parametrize('model',[IMRPhenomXP,IMRPhenomXPHM])
@pytest.mark.parametrize('version',[102,223])
def test_opposed_j_rejects_secular_twist_preserves_exact_default_carrier(model,version):
    waveform = model()
    f = t([25.,40.,80.])
    p = dict(mass1=100.,mass2=5.,spin1z=-.95,inclination=.7,phic=.3,f_ref=40.,prec_version=version)
    aligned = waveform(f,**p)
    assert all(torch.isfinite(h).all() for h in aligned)
    for convention in (0,1,5,6,7):
        with pytest.raises(ValueError,match='L \\+ S_parallel <= 0'):
            waveform(f,**p,spin1x=1e-9,convention=convention)
        if convention != 1:
            with pytest.raises(ValueError,match='L \\+ S_parallel <= 0'):
                waveform(f,**p,convention=convention)
    # Mixed batches preserve the same exact aligned row as the scalar path.
    mixed = waveform(f,mass1=t([100.,35.]),mass2=t([5.,25.]),
                     spin1x=t([0.,.2]),spin1y=t([0.,.1]),spin1z=t([-.95,.2]),
                     spin2x=t([0.,-.1]),spin2y=t([0.,.15]),spin2z=t([0.,-.1]),
                     inclination=.7,phic=.3,f_ref=40.,prec_version=version)
    for hb,hs in zip(mixed,aligned):
        torch.testing.assert_close(hb[0],hs,atol=1e-34,rtol=2e-11)


def test_opposed_j_guard_checks_reference_and_each_mode_frequency():
    p = parameters()
    p.update(m1=t(100.),m2=t(5.),s1x=t(1e-9),s1y=t(0.),s1z=t(-.95),
             s2x=t(0.),s2y=t(0.),s2z=t(0.))
    for v,v0 in [(t([.01,.02]),t(.4)),(t([.01,.4]),t(.01))]:
        with pytest.raises(ValueError,match='L \\+ S_parallel <= 0'):
            evaluate_msa_precession(**dict(p,v=v,v0=v0),prec_version=102)
    # beta=pi remains the geometry of an exactly anti-aligned J; its waveform
    # bypass is a separate nonprecessing identity, not a secular sign repair.
    exact = evaluate_msa_precession(**dict(p,v=t([.3,.4]),v0=t(.4),s1x=t(0.)),prec_version=102)
    torch.testing.assert_close(exact[1],t([math.pi,math.pi]),atol=0,rtol=0)
    scale = math.pi*105*MSUN_SEC
    f = t([.05**3/scale])
    wp = dict(mass1=100.,mass2=5.,spin1x=1e-9,spin1z=-.95,inclination=.7,f_ref=.01**3/scale)
    assert all(torch.isfinite(h).all() for h in IMRPhenomXPHM(modes=[(2,2)])(f,**wp))
    with pytest.raises(ValueError,match='L \\+ S_parallel <= 0'):
        IMRPhenomXPHM(modes=[(2,1)])(f,**wp)


def test_opposed_j_guard_does_not_reject_valid_msa_rows():
    p = parameters()
    p.update(v=t([.3,.4,.5]),v0=t(.4),m1=t(10.),m2=t(1.),
             s1x=t(.2),s1y=t(.1),s1z=t(-.6),s2x=t(.1),s2y=t(.2),s2z=t(.1))
    strict = evaluate_msa_precession(**p,prec_version=221)
    assert (strict[1] > math.pi/2).all()
    for value in strict:
        assert torch.isfinite(value).all()
    for strict_value,fallback_value in zip(strict,evaluate_msa_precession(**p,prec_version=223)):
        torch.testing.assert_close(strict_value,fallback_value,atol=0,rtol=0)


def test_experimental_ode_is_unavailable():
    with pytest.raises(ValueError,match='unvalidated prototype'):
        IMRPhenomXPHM()(t([25.,40.]),spin1x=.3,phi0=.4,precession_method='ode')
