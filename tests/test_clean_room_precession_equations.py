"""Independent Pratten arXiv:2004.06503v2 identities; no oracle imports."""
import math

import pytest
import torch

from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM, wigner_small_d_matrix
from torchwave.approximants.precession_msa import (
    evaluate_msa_precession, compute_l_3pn, compute_effective_chip,
)

DTYPE = torch.float64


def carrier_with_same_remnant(model, f, params):
    """Rotation closure compares carriers with the same remnant, including spin tilt."""
    def t(value):
        return torch.as_tensor(value, dtype=f.dtype, device=f.device)
    m1, m2 = t(params['mass1']), t(params['mass2'])
    chi1, chi2 = t(params['spin1z']), t(params['spin2z'])
    chip = compute_effective_chip(
        m1, m2, *(t(params.get(key, 0.0)) for key in ('spin1x', 'spin1y', 'spin2x', 'spin2y')),
    )
    final_spin = XASParameterFits.fit_final_spin_nnlo(m1*m2/(m1+m2)**2, chi1, chi2, chip)
    options = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                   inclination=params['inclination'], f_ref=params['f_ref'],
                   _final_spin_override=final_spin)
    if isinstance(model, IMRPhenomXP):
        return model.xas_model(f, phi_c=params['phic'], **options)
    return model.xhm_model(f, phic=params['phic'], modes=model.modes, **options)


def test_appendix_a_signed_elements_and_group_law():
    beta = torch.tensor([0.0, 0.2, 1.7, math.pi], dtype=DTYPE)
    c, s = torch.cos(beta / 2), torch.sin(beta / 2)
    d = wigner_small_d_matrix(2, beta)
    torch.testing.assert_close(d[:, 3, 4], 2*c**3*s, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(d[:, 4, 3], -2*c**3*s, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(d[:, 2, 4], math.sqrt(6)*c**2*s**2, atol=1e-15, rtol=1e-14)
    for ell in (2, 3, 4):
        d = wigner_small_d_matrix(ell, beta)
        identity = torch.eye(2*ell+1, dtype=DTYPE).expand_as(d)
        torch.testing.assert_close(d @ d.transpose(-1, -2), identity, atol=2e-14, rtol=2e-14)
        d2 = wigner_small_d_matrix(ell, beta + 0.31)
        torch.testing.assert_close(d @ wigner_small_d_matrix(ell, beta*0+0.31), d2, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('version', [101, 102])
def test_one_spin_cone_is_vector_sum_at_reference(version):
    # Eq4.4, C5 and G6: the physical in-plane spin has no A1 factor.
    def t(x): return torch.tensor(x, dtype=DTYPE)
    v0, m1, m2 = t(.24), t(40.), t(20.)
    a, b, g, theta, zeta = evaluate_msa_precession(
        v0[None], v0, m1, m2, t(.3), t(.2), t(.4),
        t(0.), t(0.), t(-.2), t(.7), t(.3), prec_version=version,
    )
    eta, delta = t(2/9), t(1/3)
    sz, sigma = t((4*.4-.2)/9), t((-.2)/3-2*.4/3)
    orbital = (IMRPhenomP.orbital_angular_momentum(v0, eta) if version == 101
               else compute_l_3pn(v0, eta, delta, sz, sigma))
    j = torch.stack((t(4*.3/9), t(4*.2/9), orbital+sz))
    expected_beta = torch.atan2(torch.linalg.vector_norm(j[:2]), j[2])
    torch.testing.assert_close(b[0], expected_beta, atol=1e-14, rtol=1e-14)
    # Verify the complete C13=C16 rotation, not just individual angles.
    def rz(x):
        c, s = torch.cos(x), torch.sin(x)
        return torch.stack((c,-s,t(0.),s,c,t(0.),t(0.),t(0.),t(1.))).reshape(3,3)
    def ry(x):
        c, s = torch.cos(x), torch.sin(x)
        return torch.stack((c,t(0.),s,t(0.),t(1.),t(0.),-s,t(0.),c)).reshape(3,3)
    rotation = rz(a[0]) @ ry(b[0]) @ rz(g[0])
    torch.testing.assert_close(rotation @ j, torch.stack((t(0.), t(0.), torch.linalg.vector_norm(j))), atol=2e-14, rtol=2e-14)
    line = torch.stack((torch.sin(t(.7))*torch.sin(t(.3)),torch.sin(t(.7))*torch.cos(t(.3)),torch.cos(t(.7))))
    torch.testing.assert_close(rotation @ line, torch.stack((torch.sin(theta),t(0.),torch.cos(theta))), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('factory', [IMRPhenomP, IMRPhenomXP, lambda: IMRPhenomXPHM(modes=[(2,2)])])
def test_quadrupole_reference_closure_label_exchange_and_dc_gradients(factory):
    model = factory()
    f = torch.tensor([0., 25., 40., 80.], dtype=DTYPE)
    sx = torch.tensor(.3, dtype=DTYPE, requires_grad=True)
    params = dict(mass1=40., mass2=20., spin1x=sx, spin1y=.2,
                  spin1z=.4, spin2z=-.2, inclination=.7, phic=.3, f_ref=40.)
    hp, hc = model(f, **params)
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    assert hp[0] == 0 and hc[0] == 0
    # Pv2 retains its inherited distinct frame mapping pending derivation.
    # XP v223 also uses distinct reference-frame and opening-beta orbital-L
    # prescriptions. Its composed-strain residual is checked against public
    # LAL separately; the strict Eq. (79) vector-rotation gate above remains.
    if not isinstance(model, (IMRPhenomP, IMRPhenomXP)):
        hp0, hc0 = carrier_with_same_remnant(model, f, params)
        for h, h0 in ((hp,hp0),(hc,hc0)):
            torch.testing.assert_close(h[2], h0[2], rtol=3e-13, atol=1e-35)
    swapped = {**params, 'mass1':20., 'mass2':40., 'spin1x':0., 'spin1y':0.,
               'spin1z':-.2, 'spin2x':sx, 'spin2y':.2, 'spin2z':.4}
    for h, other in zip((hp,hc), model(f, **swapped)):
        torch.testing.assert_close(h, other, rtol=1e-13, atol=1e-35)
    ((hp.real.square()+hc.imag.square()).sum()*1e44).backward()
    assert torch.isfinite(sx.grad)
    assert sx.grad.abs() > 0


@pytest.mark.parametrize('lm', IMRPhenomXPHM.SUPPORTED_MODES)
def test_each_mode_tends_continuously_to_aligned_projection(lm):
    # Spin-2 projection plus C4 azimuth needs i**m carrier conversion.
    f = torch.tensor([25., 40., 80.], dtype=DTYPE)
    model = IMRPhenomXPHM(modes=[lm])
    params = dict(mass1=40.,mass2=20.,spin1z=.4,spin2z=-.2,phic=.3,f_ref=40.,inclination=.7)
    for h, h0 in zip(model(f, spin1x=1e-9, **params), model(f, **params)):
        assert torch.linalg.vector_norm(h-h0)/torch.linalg.vector_norm(h0) < 1e-7


def test_scalar_masses_with_batched_spins_match_scalar_calls():
    f = torch.tensor([25., 40., 80.], dtype=DTYPE)
    for model in (IMRPhenomP(), IMRPhenomXP(), IMRPhenomXPHM(modes=[(2,2)])):
        params = dict(mass1=40.,mass2=20.,spin1z=.4,spin2z=-.2,phic=.3,f_ref=40.,inclination=.7)
        batch = model(f, spin1x=torch.tensor([.1,.3],dtype=DTYPE), **params)
        for i, sx in enumerate((.1,.3)):
            for h, hs in zip(batch, model(f, spin1x=sx, **params)):
                torch.testing.assert_close(h[i], hs, atol=1e-34, rtol=2e-12)


@pytest.mark.parametrize('lm', IMRPhenomXPHM.SUPPORTED_MODES)
def test_each_mode_reference_rotation_uses_its_spa_orbital_frequency(lm):
    # E18/E19 identify f_orb=f/mp, hence f=mp*f_ref/2 at the initial frame.
    f = torch.tensor([lm[1]*40./2],dtype=DTYPE)
    model = IMRPhenomXPHM(modes=[lm])
    params = dict(mass1=40.,mass2=20.,spin1z=.4,spin2z=-.2,phic=.3,f_ref=40.,inclination=.7)
    precessing_params = dict(params, spin1x=.3, spin1y=.2)
    for h, h0 in zip(model(f, **precessing_params),
                     carrier_with_same_remnant(model, f, precessing_params)):
        torch.testing.assert_close(h,h0,atol=1e-34,rtol=5e-12)
