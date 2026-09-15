import torch
import math
import numpy as np
from torchwave.approximants.precession_msa_pn_clean import *

def test_averages_and_moments():
    a = precession_averages(.1, .2, .5, 2/9, 1.0, .25, .04, .09, .01)
    u1 = (.1*1.5 - .5*(2/9)*.2)/((2/9)*.75)
    u2 = -.5*(.1*1.5-(2/9)*.2)/((2/9)*.75)
    corr = (.75**2*.01**2)/(32*(2/9)**2*.5**2)
    assert torch.allclose(a["s1l"], torch.tensor(u1, dtype=torch.float64))
    assert torch.allclose(a["s1l2"], torch.tensor(u1*u1+corr, dtype=torch.float64))
    assert torch.allclose(a["s2l2"], torch.tensor(u2*u2+.25*corr, dtype=torch.float64))
    assert torch.allclose(a["s1ls2l"], torch.tensor(u1*u2-.5*corr, dtype=torch.float64))
    assert torch.allclose(a["s12"], torch.tensor((.625-.04-.09)/2, dtype=torch.float64))

def test_averages_physical_uniform_cycle_projection():
    from torchwave.approximants.precession_msa_initial_clean import initial_invariants, integration_constant_c1, cubic_coefficients
    for m1, m2, z1, z2 in ((35., 25., (.2, .1, .2), (-.1, .15, -.1)),
                            (45., 15., (.45, -.2, .3), (.1, .2, -.25))):
        v0 = torch.tensor(math.pi * (m1 + m2) * 4.925490947641267e-6 * 20., dtype=torch.float64).pow(1 / 3)
        # Use the 20 Hz reference's PN orbital scale in normalized units.
        q0, eta0 = m2/m1, (m1*m2)/(m1+m2)**2
        q, eta, dm, S1, S2, L, J, xi = initial_invariants(m1, m2, z1, z2, [0., 0., float(eta0/v0)])
        B0, C0, D0 = cubic_coefficients(q, eta, dm, S1*S1, S2*S2, L, J, xi)
        roots0 = np.roots([1., float(B0), float(C0), float(D0)])
        assert np.max(np.abs(roots0.imag)) < 1e-12
        sm, sp = (torch.tensor(x, dtype=torch.float64) for x in np.sort(roots0.real)[1:])
        sav = (sp+sm)/2
        c1 = integration_constant_c1(J, eta/v0, v0, sav)
        got = precession_averages(c1, xi, q, eta, sp, sm, S1*S1, S2*S2, v0)
        phase = torch.arange(100000, dtype=torch.float64) * (torch.pi*2/100000)
        LL, J2 = eta/v0, eta**2/v0**2 + 2*c1/v0 + sav
        x = sav + (sp-sm)/2*torch.cos(2*phase)
        u1 = (q*xi-(1+q)*(J2-LL**2-x)/(2*LL))/(q**2-1)
        u2 = (J2-LL**2-x)/(2*LL)-u1
        for key, ref in (("s1l",u1.mean()), ("s2l",u2.mean()), ("s1l2",(u1*u1).mean()), ("s2l2",(u2*u2).mean()), ("s1ls2l",(u1*u2).mean())):
            assert torch.allclose(got[key], ref, rtol=1e-11, atol=1e-13)

def test_flux_reciprocal_polynomial_identity():
    a = flux_coefficients(.2, .5, .03, -.01, .002, .001, .0002, .001, .002, -.0003)
    g = reciprocal_coefficients(a)
    coeff = [a["a0"]*g["g0"], a["a0"]*g["g1"],
             a["a0"]*g["g2"]+a["a0"]*a["a2"]*g["g0"],
             a["a0"]*g["g3"]+a["a0"]*a["a3"]*g["g0"],
             a["a0"]*g["g4"]+a["a0"]*a["a2"]*g["g2"]+a["a0"]*a["a4"]*g["g0"],
             a["a0"]*g["g5"]+a["a0"]*a["a2"]*g["g3"]+a["a0"]*a["a3"]*g["g2"]+a["a0"]*a["a5"]*g["g0"]]
    assert torch.allclose(coeff[0], torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert all(torch.allclose(x, torch.tensor(0.0, dtype=torch.float64), atol=1e-12) for x in coeff[1:])

def test_psi_anchor_batch_gradient_and_identity():
    eta, q = 2/9, .5
    delta = (1-q)/(1+q)
    c1 = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    p1, p2 = psi_coefficients(eta, delta, q, .2, c1, .3, .002, .001, 2., .1)
    assert torch.allclose(p1.detach(), 3*(2*.2*eta**2-c1.detach())/(eta*delta**2))
    v = torch.tensor([.01, .02], dtype=torch.float64, requires_grad=True)
    out = anchored_psi(v, .015, 1., delta, 2., p1, p2)
    assert torch.allclose(anchored_psi(torch.tensor(.015, dtype=torch.float64), .015, 1., delta, 2., p1, p2), torch.tensor(1., dtype=torch.float64))
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(v.grad).all() and torch.isfinite(c1.grad).all()

def test_independent_physical_cubic_expansion_converges():
    from torchwave.approximants.precession_msa_initial_clean import (
        initial_invariants, cubic_coefficients,
    )
    cases = ((35., 25., (.2, .1, .2), (-.1, .15, -.1)),
             (45., 15., (.45, -.2, .3), (.1, .2, -.25)))
    for m1, m2, chi1, chi2 in cases:
        Msec = (m1 + m2) * 4.925490947641267e-6
        eta0 = m1 * m2 / (m1 + m2) ** 2
        v0 = torch.tensor(math.pi * Msec * 20., dtype=torch.float64).pow(1 / 3)
        L0 = torch.tensor(eta0, dtype=torch.float64) / v0
        q, eta, delta, S1, S2, L, J, xi = initial_invariants(
            m1, m2, chi1, chi2, [0., 0., L0])
        s1sq, s2sq = S1 * S1, S2 * S2
        B0, C0, D0 = cubic_coefficients(q, eta, delta, s1sq, s2sq, L, J, xi)
        roots0 = np.roots([1., float(B0), float(C0), float(D0)])
        assert np.max(np.abs(roots0.imag)) < 1e-12
        s3_0, sm, sp = (torch.tensor(x, dtype=torch.float64) for x in np.sort(roots0.real))
        sav = (sp + sm) / 2
        c1 = .5 * v0 * (J * J - L * L - sav)
        av = precession_averages(c1, xi, q, eta, sp, sm, s1sq, s2sq, v0)
        flux = flux_coefficients(eta, q, av["s1l"], av["s2l"], s1sq, s2sq,
                                 av["s12"], av["s1l2"], av["s2l2"], av["s1ls2l"])
        rec = reciprocal_coefficients(flux)
        psi1, psi2 = psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq,
                                      rec["g0"], rec["g2"])
        residuals = []
        for vvalue in (.04, .02, .01, .005):
            v = torch.tensor(vvalue, dtype=torch.float64)
            LL = eta / v
            JJ = torch.sqrt(eta * eta / v**2 + 2 * c1 / v + sav)
            B, C, D = cubic_coefficients(q, eta, delta, s1sq, s2sq, LL, JJ, xi)
            roots_complex = np.roots([1., float(B), float(C), float(D)])
            assert np.max(np.abs(roots_complex.imag)) < 1e-12
            roots = np.sort(roots_complex.real)
            s3, smv, spv = (torch.tensor(x, dtype=torch.float64) for x in roots)
            exact = (1 - xi * v) * torch.sqrt(spv - s3) * v / torch.sqrt(eta * delta**2)
            series = 1 + (2 / 3) * psi1 * v + (psi2 / 3 - rec["g2"] / rec["g0"]) * v**2
            residuals.append(float(torch.abs(exact - series)))
        ratios = [residuals[i] / residuals[i + 1] for i in range(3)]
        assert all(r > 7.5 for r in ratios), (residuals, ratios)
        assert residuals[-1] < 2e-7
