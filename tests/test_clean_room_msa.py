"""Checks of the published MSA defining equations; no waveform oracle."""
import math
import mpmath as mp
import pytest
import torch
from torchwave.approximants.precession_msa import (
    elliptic_f, jacobi_sn_cn_dn, msa_phase_primitives, msa_oscillatory_angles,
    msa_nutation_phase, msa_opening_angle, solve_spin_roots, evaluate_msa_precession,
    evaluate_msa_dynamics, compute_l_3pn,
)
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)


def t(x):
    return torch.tensor(x,dtype=torch.float64)


def spin_state():
    q=t(.5);eta=q/(1+q)**2;delta=(1-q)/(1+q);v=t(.22)
    s1=t([.3,.2,.1])*4/9;s2=t([-.2,.3,-.1])/9
    s1q=(s1*s1).sum();s2q=(s2*s2).sum();s0q=((s1+s2)**2).sum()
    xi=(1+q)*s1[2]+(1+1/q)*s2[2];l=eta/v
    jq=l*l+2*l*(s1[2]+s2[2])+s0q
    roots=solve_spin_roots(l,jq,s1q,s2q,xi,q,eta,delta)
    return q,eta,delta,v,s1,s2,s1q,s2q,s0q,xi,l,jq,roots


@pytest.mark.parametrize('m',[0.,.1,.7,.99])
def test_elliptic_inverse_and_defining_differential_equation(m):
    phi=torch.linspace(-1.55,1.55,31,dtype=torch.float64)
    u=elliptic_f(phi,t(m)).detach().requires_grad_()
    sn,cn,dn=jacobi_sn_cn_dn(u,t(m))
    torch.testing.assert_close(sn,phi.sin(),rtol=2e-13,atol=2e-14)
    torch.testing.assert_close(torch.autograd.grad(sn.sum(),u)[0],cn*dn,rtol=2e-13,atol=2e-14)
    torch.testing.assert_close(sn*sn+cn*cn,torch.ones_like(sn),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(dn*dn+m*sn*sn,torch.ones_like(sn),rtol=1e-14,atol=1e-14)


def test_secular_primitives_differentiate_to_equation_65():
    v=t([.13,.19,.27,.42]).requires_grad_();eta,c1,sav=t(.22),t(.03),t(.04)
    p=msa_phase_primitives(v,eta,c1,sav)
    j=(eta**2/v**2+2*c1/v+sav).sqrt()
    for n in range(6):
        derivative=torch.autograd.grad(p[:,n].sum(),v,retain_graph=True)[0]
        torch.testing.assert_close(derivative,j*v**(n-3),rtol=2e-13,atol=2e-13)


def test_oscillatory_primitives_obey_defining_rates_and_have_zero_mean():
    q,eta,delta,v,_,_,s1q,s2q,_,xi,l,jq,roots=spin_state()
    s3,sm,sp,_=roots;j=jq.sqrt()
    psi=torch.linspace(-3.,4.,77,dtype=torch.float64,requires_grad=True)
    a,e=msa_oscillatory_angles(v,j,l,xi,eta,s1q,s2q,delta,s3,sm,sp,psi)
    psidot=3/(4*eta.sqrt())*(1-xi*v)*v**6*(sp-s3).sqrt()
    # Eq. 29 is geometrical, independent of the partial-fraction implementation.
    def omega(x):
        spin_sq=sp-(sp-sm)*x.sin()**2
        den=4*jq*l*l-(jq+l*l-spin_sq)**2
        alpha=j/2*v**6*(1+3/(2*eta)*(1-xi*v)-3*(1+q)/(2*q*den)*(1-xi*v)*(4*(1-q)*l*l*(s1q-s2q)-(1+q)*(jq-l*l-spin_sq)*(jq-l*l-spin_sq-4*eta*l*xi)))
        return alpha,alpha*(jq+l*l-spin_sq)/(2*j*l)
    cycle=torch.arange(32768,dtype=torch.float64)*math.pi/32768
    averages=[o.mean() for o in omega(cycle)]
    for angle,rate,average in zip((a,e),omega(psi),averages):
        derivative=torch.autograd.grad(angle.sum(),psi,retain_graph=True)[0]
        torch.testing.assert_close(derivative*psidot,rate-average,rtol=1e-10,atol=1e-16)
    for correction in msa_oscillatory_angles(v,j,l,xi,eta,s1q,s2q,delta,s3,sm,sp,cycle):
        assert abs(correction.mean()) < 1e-13


def test_nutation_phase_has_the_cubic_small_velocity_expansion():
    q,eta,delta,v0,_,_,s1q,s2q,_,xi,l0,j0sq,roots=spin_state()
    _,sm,sp,_=roots;sav=(sm+sp)/2;c1=v0*(j0sq-l0*l0-sav)/2
    # A 1PN expansion leaves O(v^3) relative errors in dpsi/dv.
    errors=[]
    for value in (.04,.02):
        v=t(value).requires_grad_()
        psi=msa_nutation_phase(v,v0,eta,delta,q,xi,c1,sav,s1q,s2q,t(0.))
        derivative=torch.autograd.grad(psi,v)[0]
        l=eta/v;jq=l*l+2*c1/v+sav
        s3,_,sp,_=solve_spin_roots(l,jq,s1q,s2q,xi,q,eta,delta)
        g0=5/(96*eta);g2=g0*(743/336+11*eta/4)
        exact=3/(4*eta.sqrt())*(1-xi*v)*v**6*(sp-s3).sqrt()*3*(g0+g2*v*v)/v**9
        errors.append(abs((derivative-exact)/exact).item())
    assert errors[1] < errors[0]/5


def test_msa_reference_geometry_gradient_and_grid_independence():
    v=t([.18,.22,.3,.4]).requires_grad_()
    args=[t(.22),t(40.),t(20.),t(.3),t(.2),t(.1),t(-.2),t(.3),t(-.1),t(.7),t(.3)]
    result=evaluate_msa_precession(v,*args,223)
    subset=evaluate_msa_precession(v[[3,0,1]],*args,223)
    for full,part in zip(result[:3],subset[:3]):
        torch.testing.assert_close(full[[3,0,1]],part,rtol=1e-13,atol=1e-13)
    assert torch.isfinite(torch.autograd.grad(sum(x.sum() for x in result[:3]),v)[0]).all()
    # The PN opening angle follows the precession-averaged J(L) relation.
    # Only its Newtonian limit is exactly the supplied reference-spin cone.
    from torchwave.approximants.precession_msa import compute_l_3pn
    _,eta,_,v0,_,_,_,_,s0q,_,l0,j0sq,roots=spin_state()
    _,sm,sp,_=roots;sav=(sm+sp)/2;c1=v0*(j0sq-l0*l0-sav)/2
    sz=t(4*.1/9-.1/9);sigma=t(-.1/3-2*.1/3)
    orbital=compute_l_3pn(t(.22),t(2/9),t(1/3),sz,sigma)
    jpn=(orbital*orbital+2*c1*orbital/eta+sav).sqrt()
    expected=((jpn*jpn+orbital*orbital-s0q)/(2*jpn*orbital)).acos()
    torch.testing.assert_close(result[1][1],expected,rtol=1e-12,atol=1e-13)
    newtonian=msa_opening_angle(l0,eta,c1,sav,s0q)
    supplied=torch.atan2(t(math.hypot((4*.3-.2)/9,(4*.2+.3)/9)),l0+sz)
    torch.testing.assert_close(newtonian,supplied,rtol=1e-12,atol=1e-13)


def test_q10_msa_matches_clean_published_equations_before_compatibility_split():
    """The failing full-waveform holdout is not a native MSA transcription error."""
    f = torch.linspace(20., 220., 401, dtype=torch.float64)
    m1, m2 = t(60.), t(6.)
    chi1, chi2 = t([.2, .1, .8]), t([.1, 0., .1])
    total = m1 + m2
    q = m2 / m1
    eta = m1 * m2 / total.square()
    delta = (m1 - m2) / total
    n1, n2 = m1 / total, m2 / total
    spin1, spin2 = n1.square() * chi1, n2.square() * chi2
    v = (math.pi * total * 4.925490947641267e-6 * f).pow(1 / 3)
    v0 = v[0]
    s_ell = spin1[2] + spin2[2]
    sigma_ell = n2 * chi2[2] - n1 * chi1[2]
    l_pn = compute_l_3pn(v, eta, delta, s_ell, sigma_ell)

    alpha, beta, gamma, valid = evaluate_msa_dynamics(
        v, v0, eta, delta, q, spin1, spin2, l_pn,
    )
    equation = assemble_precession_msa_angles(
        f, m1, m2, chi1, chi2, f_ref=20., compatibility="equation",
        beta_pn=True, beta_pn_version=102,
    )
    torch.testing.assert_close(alpha, equation["alpha"], rtol=0, atol=3e-10)
    torch.testing.assert_close(gamma, equation["gamma"], rtol=0, atol=3e-10)
    torch.testing.assert_close(beta.cos(), equation["cos_beta_pn"], rtol=0, atol=1e-10)
    assert bool(valid)

    # The separately audited v223 compatibility path first makes a material choice
    # after the common roots/c1/<S^2>: it uses initial-reference rather than
    # precession-averaged spin contractions in the published flux polynomial.
    compatibility = assemble_precession_msa_angles(
        f, m1, m2, chi1, chi2, f_ref=20., compatibility="v223",
    )
    for key in ("roots_reference", "c1", "sav"):
        torch.testing.assert_close(equation[key], compatibility[key], rtol=0, atol=0)
    for key in (
        "mean_s1_l", "mean_s2_l", "mean_s1_l2", "mean_s2_l2",
        "mean_s1ls2l",
    ):
        torch.testing.assert_close(
            equation[key], compatibility[key], rtol=1e-14, atol=1e-18,
        )
    flux_delta = abs(
        equation["flux_couplings"]["s1l"]
        - compatibility["flux_couplings"]["s1l"]
    )
    assert flux_delta > 5e-4
    assert abs(equation["psi2"] - compatibility["psi2"]) > 2


def test_q10_msa_flux_spin_couplings_obey_published_scalar_averages():
    """Appendix A (90)--(95) fixes the flux inputs independently of Torch.

    The MSA roots and c1 are upstream inputs to this isolated scalar gate;
    their separate defining-equation checks live above.  In particular,
    initial spin projections cannot replace the averaged flux projections.
    """
    m1, m2 = 60., 6.
    chi1, chi2 = (.2, .1, .8), (.1, 0., .1)
    f_ref = 20.
    equation = assemble_precession_msa_angles(
        t([f_ref]), t(m1), t(m2), t(chi1), t(chi2),
        f_ref=f_ref, compatibility="equation",
    )
    compatibility = assemble_precession_msa_angles(
        t([f_ref]), t(m1), t(m2), t(chi1), t(chi2),
        f_ref=f_ref, compatibility="v223",
    )

    with mp.workdps(50):
        def scalar(value):
            return mp.mpf(float(value))
        q = mp.mpf(1) / 10
        eta = q / (1 + q)**2
        n1, n2 = 1 / (1 + q), q / (1 + q)
        s1 = [n1**2 * scalar(component) for component in chi1]
        s2 = [n2**2 * scalar(component) for component in chi2]
        s1sq = sum(component**2 for component in s1)
        s2sq = sum(component**2 for component in s2)
        xi = (1 + q) * s1[2] + (1 + 1/q) * s2[2]
        c1 = scalar(equation["c1"])
        roots = equation["roots_reference"]
        sminus2, splus2 = scalar(roots[1]), scalar(roots[2])
        sav = (sminus2 + splus2) / 2
        v0 = (mp.pi * (m1 + m2) * mp.mpf("4.925490947641267e-6") * f_ref)**(mp.mpf(1)/3)

        mean1 = (c1 * (1 + q) - q * eta * xi) / (eta * (1 - q**2))
        mean2 = -q * (c1 * (1 + q) - eta * xi) / (eta * (1 - q**2))
        variance = (splus2 - sminus2)**2 * v0**2 / (32 * eta**2 * (1 - q)**2)
        published = {
            "s1l": mean1,
            "s2l": mean2,
            "s12": (sav - s1sq - s2sq) / 2,
            "s1l2": mean1**2 + variance,
            "s2l2": mean2**2 + q**2 * variance,
            "s1ls2l": mean1 * mean2 - q * variance,
        }
        for key, expected in published.items():
            assert math.isclose(float(equation["flux_couplings"][key]),
                                float(expected), rel_tol=2e-12, abs_tol=2e-15), key

        # The initial-reference compatibility recipe makes a materially
        # different choice at this first flux boundary for the frozen case.
        assert abs(float(compatibility["flux_couplings"]["s1l"])
                   - float(published["s1l"])) > 5e-4


def test_q10_msa_appendix_c_quadratic_phase_diagnostic():
    """The printed Appendix C (117)--(118) does not select either phase path.

    This is a diagnostic of the frozen q=10 case, not a waveform acceptance
    threshold. The independently checked roots and c1 are common inputs.
    """
    m1, m2 = 60., 6.
    chi1, chi2 = (.2, .1, .8), (.1, 0., .1)
    args = (t([20.]), t(m1), t(m2), t(chi1), t(chi2))
    equation = assemble_precession_msa_angles(
        *args, f_ref=20., compatibility="equation",
    )
    compatibility = assemble_precession_msa_angles(
        *args, f_ref=20., compatibility="v223",
    )

    # Chatziioannou et al., arXiv:1703.03967v3, Appendix C (116)--(118).
    with mp.workdps(50):
        def scalar(value):
            return mp.mpf(float(value))
        q = mp.mpf(1) / 10
        eta = q / (1 + q)**2
        delta_m = (1 - q) / (1 + q)
        n1, n2 = 1 / (1 + q), q / (1 + q)
        s1 = [n1**2 * scalar(component) for component in chi1]
        s2 = [n2**2 * scalar(component) for component in chi2]
        s1sq = sum(component**2 for component in s1)
        s2sq = sum(component**2 for component in s2)
        xi = (1 + q) * s1[2] + (1 + 1/q) * s2[2]
        c1, sav = scalar(equation["c1"]), scalar(equation["sav"])
        psi1 = 3 * (2 * xi * eta**2 - c1) / (eta * delta_m**2)
        first = (c1**2 * eta / (q * delta_m**4)
                 - 2 * c1 * eta**3 * (1 + q) * xi / (q * delta_m**4)
                 - eta**2 * (delta_m**2 * s1sq - eta**2 * xi**2) / delta_m**4)
        second = (c1**2 * eta**2 / delta_m**4
                  - 2 * c1 * eta**3 * (1 + q) * xi / delta_m**4
                  - eta**2 * (delta_m**2 * s2sq - eta**2 * xi**2) / delta_m**4)
        root_delta = mp.sqrt(first * second)
        g2_over_g0 = mp.mpf(743) / 336 + 11 * eta / 4
        psi2 = 3 * g2_over_g0 + 3 / (2 * eta**3) * (
            2 * root_delta - 2 * eta**2 * sav / delta_m**2
            - 10 * eta * c1**2 / delta_m**4
            + 2 * eta**2 * (7 + 6*q + 7*q**2) * c1 * xi
              / (delta_m**2 * (1-q)**2)
            - eta**3 * (3 + 4*q + 3*q**2) * xi**2
              / (delta_m**2 * (1-q)**2)
            + eta * (q * (2+q) * s1sq + (1+2*q) * s2sq) / (1-q)**2
        )

        assert math.isclose(float(psi1), float(equation["psi1"]),
                            rel_tol=2e-12, abs_tol=2e-12)
        assert first < 0 and second < 0  # fixes the real square-root branch
        assert abs(float(psi2 - scalar(equation["psi2"]))) > .25
        assert abs(float(psi2 - scalar(compatibility["psi2"]))) > 2.


def test_opening_angle_uses_precession_averaged_j_at_any_nutation_phase():
    # Recover J from the triangle and check the defining averaged dJ/dL.
    l=t([.6,1.,1.4]).requires_grad_();eta,c1,sav=t(.22),t(.02),t(.04)
    for spin_sq in (t(.025),t(.055)):
        beta=msa_opening_angle(l,eta,c1,sav,spin_sq)
        j=l*beta.cos()+(spin_sq-l*l*beta.sin()**2).sqrt()
        expected=(l*l+2*c1*l/eta+sav).sqrt()
        torch.testing.assert_close(j,expected,rtol=1e-13,atol=1e-13)
        derivative=torch.autograd.grad(j.sum(),l,retain_graph=True)[0]
        torch.testing.assert_close(derivative,(j*j+l*l-sav)/(2*j*l),rtol=1e-13,atol=1e-13)


def test_leading_oscillatory_correction_is_smooth_and_pi_periodic():
    _,eta,delta,v,_,_,s1q,s2q,_,xi,l,jq,roots=spin_state()
    s3,sm,sp,_=roots
    psi=t([-math.pi,-math.pi/2,0.,math.pi/2,math.pi]).requires_grad_()
    def corrections(x):
        return msa_oscillatory_angles(v,jq.sqrt(),l,xi,eta,s1q,s2q,delta,s3,sm,sp,x)
    for value,shifted in zip(corrections(psi),corrections(psi+math.pi)):
        torch.testing.assert_close(value,shifted,rtol=1e-13,atol=1e-13)
        assert torch.isfinite(torch.autograd.grad(value.sum(),psi,retain_graph=True)[0]).all()
    # The elliptic spin solution's distinct 2K(m) period is retained.
    m=(sp-sm)/(sp-s3);k=elliptic_f(t(math.pi/2),m)
    u=t([.2,.7,1.2]);sn,_,_=jacobi_sn_cn_dn(u,m)
    shifted,_,_=jacobi_sn_cn_dn(u+2*k,m)
    torch.testing.assert_close(sn.square(),shifted.square(),rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('version',[220,223])
def test_degenerate_initialization_uses_nnlo_fallback(version):
    args=[t(.22),t(40.),t(20.),t(.3),t(.2),t(.1),t(0.),t(0.),t(0.),t(.7),t(.3)]
    v=t([.18,.22,.3])
    for actual,expected in zip(evaluate_msa_precession(v,*args,version),evaluate_msa_precession(v,*args,102)):
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)


def test_mixed_binary_batch_and_parameter_gradients():
    # Include regular two-spin, equal-mass, one-spin and aligned systems.
    params=t([[40,20,.3,.2,.1,-.2,.3,-.1],
              [30,30,.3,.2,.1,-.2,.3,-.1],
              [40,20,.3,.2,.1,0,0,0],
              [40,20,0,0,.1,0,0,-.1]]).requires_grad_()
    v=t([.18,.22,.3])
    result=evaluate_msa_precession(v,t(.22),*params.T.unsqueeze(-1),t(.7),t(.3),223)
    for i,p in enumerate(params):
        scalar=evaluate_msa_precession(v,t(.22),*p.unbind(),t(.7),t(.3),223)
        for batched,one in zip(result,scalar):
            torch.testing.assert_close(batched[i].squeeze(),one,rtol=3e-12,atol=3e-12)
    gradient=torch.autograd.grad(sum(x.sum() for x in result),params)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0].abs().sum() > 0


@pytest.mark.parametrize('version',[221,222])
def test_msa_without_fallback_rejects_degenerate_initialization(version):
    args=[t(.22),t(40.),t(20.),t(.3),t(.2),t(.1),t(0.),t(0.),t(0.),t(.7),t(.3)]
    with pytest.raises(ValueError,match='degenerate'):
        evaluate_msa_precession(t([.18,.22,.3]),*args,version)


def test_nonreal_secular_coefficients_trigger_whole_binary_fallback():
    # Real cubic roots do not suffice: Appendix D6 also needs cp*cm>0.
    args=[t(.22),t(20.),t(10.),t(0.),t(-.13),t(-.25),t(.2),t(.52),t(.13),t(.7),t(.3)]
    v=t([.12,.22,.32,.45])
    for actual,expected in zip(evaluate_msa_precession(v,*args,223),evaluate_msa_precession(v,*args,102)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='degenerate'):
        evaluate_msa_precession(v,*args,222)


@pytest.mark.parametrize('model_name',['xp','xphm'])
def test_method_selects_default_version_and_explicit_version_takes_precedence(model_name):
    from torchwave.approximants.imrphenom_xp import IMRPhenomXP
    from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
    model=IMRPhenomXP() if model_name=='xp' else IMRPhenomXPHM(modes=[(2,2)])
    f=t([25.,40.,80.])
    args=dict(mass1=40,mass2=20,spin1x=.3,spin1y=.2,spin1z=.1,
              spin2x=-.2,spin2y=.3,spin2z=-.1,inclination=.7,phic=.3,f_ref=40.)
    default=model(f,**args)
    nnlo=model(f,**args,precession_method='nnlo',prec_version=102)
    msa=model(f,**args,precession_method='msa')
    explicit=model(f,**args,prec_version=223)
    for a,b,c,d in zip(default,nnlo,msa,explicit):
        torch.testing.assert_close(a,c,rtol=0,atol=0)
        torch.testing.assert_close(c,d,rtol=0,atol=0)
        assert torch.linalg.vector_norm(b-a)/torch.linalg.vector_norm(a) > 1e-3
    with pytest.raises(ValueError,match='Unsupported precession version'):
        model(f,**args,prec_version=999)
    with pytest.raises(ValueError,match='precession_method'):
        model(f,**args,precession_method='invalid')


def test_four_pn_orbital_angular_momentum_has_kerr_linear_spin_limit():
    from torchwave.approximants.precession_msa import compute_l_4pn
    # Expanding exact Kerr circular L at fixed v gives SO terms
    # -10/3*v^3 - 7*v^5 - 81/4*v^7. This also catches G10h's typo.
    eta=t(1e-10);delta=(1-4*eta).sqrt();m1=(1+delta)/2
    chi=t(.3).requires_grad_();v=t([.1,.2,.3])
    l=compute_l_4pn(v,eta,delta,m1*m1*chi,-m1*chi,chi,t(0.))
    derivative=torch.autograd.grad((v*l/eta).sum(),chi)[0]
    expected=(-10/3*v**3-7*v**5-81/4*v**7).sum()
    torch.testing.assert_close(derivative,expected,rtol=1e-9,atol=1e-13)
