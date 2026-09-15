import torch
import pytest
from torchwave.approximants.precession_msa_oscillatory_clean import oscillatory_correction, rational_coefficients
from torchwave.approximants.precession_msa_initial_clean import initial_invariants, cubic_coefficients, ordered_roots


def _vals(psi):
    # Distinct real factors 1-lambda sin^2(psi), away from repeated roots.
    return dict(psi=psi, J=torch.tensor(1.7, dtype=torch.float64), psidot=torch.tensor(.23, dtype=torch.float64),
                a=torch.tensor(.11, dtype=torch.float64), c0=torch.tensor(.37, dtype=torch.float64),
                c2=torch.tensor(-.19, dtype=torch.float64), c4=torch.tensor(.06, dtype=torch.float64),
                d0=torch.tensor(1., dtype=torch.float64), d2=torch.tensor(-.8, dtype=torch.float64),
                d4=torch.tensor(.1, dtype=torch.float64), atheta=torch.tensor(.8, dtype=torch.float64),
                btheta=torch.tensor(.2, dtype=torch.float64))


def test_oscillatory_rates_are_mean_zero():
    psi = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    v = _vals(psi); phi, zeta = oscillatory_correction(**v)
    dphi = torch.autograd.grad(phi, psi, retain_graph=True)[0]
    dzeta = torch.autograd.grad(zeta, psi)[0]
    x = torch.sin(psi)**2
    r = (v['c0'] + v['c2']*x + v['c4']*x*x)/(v['d0'] + v['d2']*x + v['d4']*x*x)
    disc = v['d2']**2-4*v['d0']*v['d4']; root=torch.sqrt(disc)
    l1=(-v['d2']+root)/(2*v['d0']); l2=(-v['d2']-root)/(2*v['d0'])
    r0=v['c4']/v['d4']; u=v['c0']/v['d0']-r0; w=v['c2']/v['d0']+r0*(l1+l2)
    aa=(w+u*l1)/(l1-l2); bb=(-w-u*l2)/(l1-l2)
    mean_r=r0+aa/torch.sqrt(1-l1)+bb/torch.sqrt(1-l2)
    expected_phi=v['J']*(r-mean_r)
    k1, k2 = torch.sqrt(1-l1), torch.sqrt(1-l2)
    mean_z=(v['a']+r0)*(v['atheta']+v['btheta']/2) + aa*(v['atheta']/k1 + v['btheta']*(1/k1-1)/l1) + bb*(v['atheta']/k2 + v['btheta']*(1/k2-1)/l2)
    expected_z=v['J']*((v['a']+r)*(v['atheta']+v['btheta']*x) - mean_z)
    assert torch.allclose(dphi*v['psidot'], expected_phi, rtol=1e-8, atol=1e-9)
    assert torch.allclose(dzeta*v['psidot'], expected_z, rtol=1e-8, atol=1e-9)


def test_periodicity_batch_and_gradient():
    psi=torch.tensor([.2,.7,1.4],dtype=torch.float64,requires_grad=True)
    v=_vals(psi); phi,z=oscillatory_correction(**v)
    phi2,z2=oscillatory_correction(**_vals(psi+torch.pi))
    assert torch.allclose(phi,phi2,atol=1e-12) and torch.allclose(z,z2,atol=1e-12)
    assert torch.isfinite(torch.autograd.grad((phi+z).sum(),psi)[0]).all()


def test_dense_cycle_means_and_physical_root_domain():
    # The requested 35/25 two-spin initial state gives the physical negative
    # third root; dense quadrature independently checks the cycle means.
    Msec = 60 * 4.925490947641267e-6
    v = torch.tensor(torch.pi * Msec * 20, dtype=torch.float64).pow(1/3)
    orbital = torch.tensor([0., 0., (35*25/60**2)/v], dtype=torch.float64)
    q, eta, delta, S1, S2, L, J, xi = initial_invariants(
        35., 25., [.2, .1, .2], [-.1, .15, -.1], orbital)
    B, C, D = cubic_coefficients(q, eta, delta, S1*S1, S2*S2, L, J, xi)
    s3, sm, sp = ordered_roots(B, C, D)
    vals = rational_coefficients(v, eta, xi, torch.tensor(.06, dtype=torch.float64), S1*S1, S2*S2,
        delta, sp, sm, s3, J=J, L=L)
    J,a,c0,c2,c4,d0,d2,d4,psidot = vals
    grid = torch.linspace(0, torch.pi, 200001, dtype=torch.float64)
    x = torch.sin(grid)**2
    rate = (c0+c2*x+c4*x*x)/(d0+d2*x+d4*x*x)
    mean_r = torch.trapezoid(rate, grid)/torch.pi
    atheta0, btheta0 = (J**2 + L**2 - sp)/(2*J*L), (sp-sm)/(2*J*L)
    mean_z = torch.trapezoid((a+rate)*(atheta0+btheta0*x), grid)/torch.pi
    psi = torch.tensor(.41, dtype=torch.float64, requires_grad=True)
    atheta, btheta = atheta0, btheta0
    out = dict(psi=psi, J=J, psidot=psidot, a=a, c0=c0, c2=c2, c4=c4,
               d0=d0, d2=d2, d4=d4, atheta=atheta, btheta=btheta)
    phi,z = oscillatory_correction(**out)
    dr = torch.autograd.grad(phi, psi, retain_graph=True)[0]*psidot/J
    dz = torch.autograd.grad(z, psi)[0]*psidot/J
    xx=torch.sin(psi)**2; rr=(out['c0']+out['c2']*xx+out['c4']*xx*xx)/(out['d0']+out['d2']*xx+out['d4']*xx*xx)
    assert abs(float((dr-(rr-mean_r)).detach())) < 2e-8
    assert abs(float((dz-((out['a']+rr)*(atheta+btheta*xx)-mean_z)).detach())) < 2e-8


def test_physical_root_domain_rejects_unordered_roots():
    args = (torch.tensor(.3, dtype=torch.float64), torch.tensor(.24, dtype=torch.float64),
            torch.tensor(.08, dtype=torch.float64), torch.tensor(.06, dtype=torch.float64),
            torch.tensor(.20, dtype=torch.float64), torch.tensor(.10, dtype=torch.float64),
            torch.tensor(.20, dtype=torch.float64), torch.tensor(-.10, dtype=torch.float64),
            torch.tensor(-.20, dtype=torch.float64), torch.tensor(-.20, dtype=torch.float64))
    with pytest.raises(ValueError, match="invalid nondegenerate"):
        rational_coefficients(*args, J=torch.tensor(1.2, dtype=torch.float64), L=torch.tensor(1.0, dtype=torch.float64))
