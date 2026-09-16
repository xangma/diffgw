import torch
import pytest

from torchwave.approximants.precession_msa_secular_clean import (
    angular_momentum, phi_z_primitives, zeta_primitives, anchored_angles,
    omega_coefficients, averaged_coefficients,
)


def _case():
    return dict(eta=torch.tensor(.24, dtype=torch.float64), xi=torch.tensor(.08, dtype=torch.float64),
                c1=torch.tensor(.06, dtype=torch.float64), s1sq=torch.tensor(.20, dtype=torch.float64),
                s2sq=torch.tensor(.10, dtype=torch.float64), delta=torch.tensor(.20, dtype=torch.float64),
                sp=torch.tensor(.50, dtype=torch.float64), sm=torch.tensor(.25, dtype=torch.float64),
                sav=torch.tensor(.375, dtype=torch.float64), g=[torch.tensor(x, dtype=torch.float64)
                for x in (1., 0., .2, .1, .05, .02)])


def test_phi_primitives_published_derivatives():
    c = _case(); v = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    got = phi_z_primitives(v, c['eta'], c['c1'], c['sav'])
    J = angular_momentum(v, c['eta'], c['c1'], c['sav'])
    expected = J * v ** (torch.arange(6, dtype=torch.float64) - 3)
    for n in range(6):
        d = torch.autograd.grad(got[n], v, retain_graph=True)[0]
        assert torch.allclose(d, expected[n], rtol=3e-10, atol=3e-11)


def test_zeta_primitives_and_assembly_derivative():
    c = _case(); v = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    z = zeta_primitives(v, c['eta'])
    for n in range(6):
        d = torch.autograd.grad(z[n], v, retain_graph=True)[0]
        expected = c['eta'] * (n - 3) * v ** (n - 4) if n != 3 else c['eta'] / v
        assert torch.allclose(d, expected, rtol=1e-12, atol=1e-12)
    phi_c, zet_c = averaged_coefficients(c['eta'], c['xi'], c['c1'], c['s1sq'], c['s2sq'], c['delta'], c['sp'], c['sm'], c['g'])
    p, zeta = anchored_angles(v, torch.tensor(.25, dtype=torch.float64), phi_c, zet_c, c['eta'], c['c1'], c['sav'])
    dp = torch.autograd.grad(p, v, retain_graph=True)[0]
    dz = torch.autograd.grad(zeta, v)[0]
    pp = phi_z_primitives(v, c['eta'], c['c1'], c['sav'])
    zz = zeta_primitives(v, c['eta'])
    assert torch.allclose(dp, (pp.new_tensor(phi_c) * (angular_momentum(v, c['eta'], c['c1'], c['sav']) * v ** (torch.arange(6, dtype=torch.float64)-3))).sum(), rtol=1e-9, atol=1e-10)
    assert torch.allclose(dz, (zet_c * torch.stack([c['eta'] * (n-3) * v**(n-4) if n != 3 else c['eta']/v for n in range(6)])).sum(), rtol=1e-9, atol=1e-10)


def test_anchor_batch_dtype_and_gradients():
    c = _case(); v0 = torch.tensor(.25, dtype=torch.float64)
    v = torch.tensor([.26, .31, .38], dtype=torch.float64, requires_grad=True)
    pc, zc = averaged_coefficients(c['eta'], c['xi'], c['c1'], c['s1sq'], c['s2sq'], c['delta'], c['sp'], c['sm'], c['g'])
    p, z = anchored_angles(v, v0, pc, zc, c['eta'], c['c1'], c['sav'])
    assert p.shape == v.shape and z.shape == v.shape and torch.isfinite(p).all() and torch.isfinite(z).all()
    p0, z0 = anchored_angles(v0, v0, pc, zc, c['eta'], c['c1'], c['sav'])
    assert torch.equal(p0, torch.zeros_like(p0)) and torch.equal(z0, torch.zeros_like(z0))
    assert torch.isfinite(torch.autograd.grad((p*p + z*z).sum(), v)[0]).all()
    assert phi_z_primitives(v.float(), c['eta'].float(), c['c1'].float(), c['sav'].float()).dtype == torch.float64


def test_parameter_gradient_is_scale_stable():
    c = _case(); eta = torch.tensor(.24, dtype=torch.float64, requires_grad=True)
    v = torch.tensor([.27, .32, .39], dtype=torch.float64)
    pc, zc = averaged_coefficients(eta, c['xi'], c['c1'], c['s1sq'], c['s2sq'], c['delta'], c['sp'], c['sm'], c['g'])
    value = sum(anchored_angles(v, .25, pc, zc, eta, c['c1'], c['sav'])).sum()
    analytic = torch.autograd.grad(value, eta)[0]
    h = 1e-6
    def eval_at(x):
        a, b = averaged_coefficients(x, c['xi'], c['c1'], c['s1sq'], c['s2sq'], c['delta'], c['sp'], c['sm'], c['g'])
        return sum(anchored_angles(v, .25, a, b, x, c['c1'], c['sav'])).sum()
    numeric = (eval_at(eta.detach()+h) - eval_at(eta.detach()-h)) / (2*h)
    assert torch.isfinite(analytic) and torch.isfinite(numeric)
    assert abs((analytic-numeric).item()) / max(abs(analytic.item()), 1e-8) < 3e-5


def test_omega_coefficients_reject_invalid_root_domain():
    c = _case()
    with pytest.raises(ValueError):
        omega_coefficients(c['eta'], c['xi'], c['c1'], c['s1sq'], c['s2sq'], c['delta'],
                           torch.tensor(-.5, dtype=torch.float64), c['sm'])


@pytest.mark.parametrize("eta", [torch.tensor(.24, dtype=torch.float64),
                                  torch.tensor(.1875, dtype=torch.float64)])
def test_omega_leading_baseline_matches_instantaneous_equation(eta):
    """The secular baseline must have the full equation's small-v limit.

    Eq. (phizdot) gives a/v**6 -> 1/2 + 3/(4*eta), independently of spin
    roots and of the cycle-average rational correction ``ad``.  Checking the
    decomposition ``Omega_z,0 = a1 + ad`` keeps this physical limit separate
    from the Appendix coefficient algebra.
    """
    c = _case()
    c['eta'] = eta
    oz, _ = omega_coefficients(c['eta'], c['xi'], c['c1'], c['s1sq'], c['s2sq'],
                               c['delta'], c['sp'], c['sm'])
    cp = c['sp'] * eta**2 - c['c1']**2
    cm = c['sm'] * eta**2 - c['c1']**2
    root = torch.sqrt(cp * cm)
    ad = (-3 * (c['s1sq'] - c['s2sq']) * eta * c['delta']
          + 3 * c['c1'] / eta * (c['c1'] - 2 * c['xi'] * eta**2)) / (4 * root)
    a_over_v6 = .5 + 3 / (4 * eta)
    assert torch.allclose(oz[0] - ad, a_over_v6, rtol=1e-14, atol=1e-14)
