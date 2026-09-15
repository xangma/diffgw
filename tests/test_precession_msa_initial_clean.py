import torch

from torchwave.approximants.precession_msa_initial_clean import (
    cubic_coefficients, initial_invariants, initial_phase, jacobi_sn,
    spin_oscillation, spin_average_exact, integration_constant_c1,
    averaged_spin_couplings, ordered_roots,
)


def test_invariants_ordering_and_cubic_factorization():
    inv = initial_invariants(20., 30., [0., 0., .2], [0., 0., -.1], [0., 0., 2.])
    assert torch.isclose(inv[0], torch.tensor(2 / 3, dtype=torch.float64))
    b, c, d = cubic_coefficients(2/3, 2/9, 1/3, .01, .0025, 2., 2.1, .1)
    roots = ordered_roots(torch.tensor(-1.2, dtype=torch.float64), torch.tensor(-.15, dtype=torch.float64), torch.tensor(.026, dtype=torch.float64))
    assert torch.isfinite(torch.stack((b,c,d))).all()
    bb, cc, dd = (torch.tensor(x, dtype=torch.float64) for x in (-1.2, -.15, .026))
    assert torch.allclose(roots**3 + bb*roots**2 + cc*roots + dd,
                          torch.zeros(3, dtype=torch.float64), atol=1e-10)


def test_jacobi_initial_state_and_phase_branch():
    plus, minus, third = torch.tensor(1.0), torch.tensor(.25), torch.tensor(-.5)
    target = torch.tensor(.6)
    p = initial_phase(target, plus, minus, third, torch.tensor(1.))
    assert torch.allclose(spin_oscillation(p, plus, minus, third), target, atol=1e-7)
    pn = initial_phase(target, plus, minus, third, torch.tensor(-1.))
    assert torch.allclose(jacobi_sn(pn, (plus-minus)/(plus-third)),
                          -jacobi_sn(p, (plus-minus)/(plus-third)), atol=1e-7)


def test_elliptic_path_has_finite_intrinsic_gradient():
    u = torch.tensor([.1, .3], dtype=torch.float64, requires_grad=True)
    m = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    loss = jacobi_sn(u, m).square().sum()
    loss.backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(m.grad)


def test_exact_average_and_c1_identity():
    p, n, s3 = (torch.tensor(x, dtype=torch.float64) for x in (1.0, .25, -.5))
    av = spin_average_exact(p, n, s3)
    # Eq. (26) agrees with direct period quadrature of the Jacobi solution.
    m = (p - n) / (p - s3)
    import scipy.special
    K = float(__import__('torchwave.approximants.precession_msa_initial_clean',
                        fromlist=['elliptic_K_agm']).elliptic_K_agm(m))
    u = torch.linspace(0, 4 * K, 2001, dtype=torch.float64)
    sn = torch.as_tensor(scipy.special.ellipj(u.numpy(), float(m))[0])
    samples = p + (n - p) * sn.square()
    assert torch.allclose(av, samples.mean(), atol=5e-4)
    J, L, v = (torch.tensor(x, dtype=torch.float64) for x in (1.7, 1.2, .03))
    c1 = integration_constant_c1(J, L, v, av)
    assert torch.allclose(L * L + 2 * c1 / v + av, J * J, atol=1e-12)


def test_averaged_couplings_identity_and_domain():
    out = averaged_spin_couplings(.1, -.2, 1., .25, .5, .2, .03)
    assert len(out) == 3 and all(torch.isfinite(x) for x in out)
    try:
        averaged_spin_couplings(.1, -.2, 1., .25, 1., .25, .03)
    except ValueError:
        pass
    else:
        raise AssertionError("equal-mass singular average must be rejected")


def test_direct_cubic_physical_mass_batches():
    cases = [(5., 1.4, .2, 0.), (7., 1.4, .4, 0.)]
    rows = []
    for m1, m2, z1, z2 in cases:
        q, eta, dm, s1, s2, L, J, xi = initial_invariants(
            m1, m2, [0., 0., z1], [0., 0., z2], [0., 0., 2.])
        rows.append(torch.stack((q, eta, dm, s1*s1, s2*s2, L, J, xi)))
    r = torch.stack(rows)
    B, C, D = cubic_coefficients(*[r[:, i] for i in range(8)])
    assert torch.isfinite(torch.stack((B, C, D))).all()


def test_cubic_root_gradient_matches_finite_difference():
    B = torch.tensor(-1.2, dtype=torch.float64, requires_grad=True)
    C = torch.tensor(-.15, dtype=torch.float64)
    D = torch.tensor(.026, dtype=torch.float64)
    loss = ordered_roots(B, C, D)[0]
    loss.backward()
    h = 1e-6
    fd = (ordered_roots(B.detach()+h, C, D)[0] - ordered_roots(B.detach()-h, C, D)[0])/(2*h)
    assert torch.allclose(B.grad, fd, atol=2e-8, rtol=2e-7)


def test_jacobi_period_grid_and_derivatives():
    import numpy as np
    import scipy.special
    for mv in (.001, .2, .5, .9, .99):
        m = torch.tensor(mv, dtype=torch.float64)
        u = torch.linspace(-50, 50, 1001, dtype=torch.float64)
        got = jacobi_sn(u, m).detach().numpy()
        ref = scipy.special.ellipj(u.numpy(), mv)[0]
        assert np.max(np.abs(got-ref)) < 1e-10
    for uv in (0., .37, 2.0):
        u = torch.tensor(uv, dtype=torch.float64, requires_grad=True)
        m = torch.tensor(.2, dtype=torch.float64)
        jacobi_sn(u, m).backward()
        sn, cn, dn, _ = scipy.special.ellipj(uv, .2)
        assert abs(float(u.grad) - cn*dn) < 1e-9


def test_root_domain_and_zero_modulus_gradient():
    try:
        ordered_roots(0., 1., 0.)
    except ValueError:
        pass
    else:
        raise AssertionError("one-real-root cubic must be rejected")
    p = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    av = spin_average_exact(p, torch.tensor(1., dtype=torch.float64), torch.tensor(-.2, dtype=torch.float64))
    av.backward()
    assert torch.isfinite(av) and torch.isfinite(p.grad)
