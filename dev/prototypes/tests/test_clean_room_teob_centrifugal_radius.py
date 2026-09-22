"""Clean-room checks for the TEOBResumS centrifugal-radius map.

The equations are Nagar et al., arXiv:1806.01772v2, Eqs. (23)--(25):
the BBH effective Kerr spin, and the quadratic C_Q extension for neutron
stars.  These tests do not import target-library code or outputs.
"""

import torch

from torchwave.approximants.teobresums import TEOBResumSHamiltonian


def t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def test_centrifugal_radius_matches_published_quadratic_cq_map():
    nu = t([0.20, 0.24])
    chi1, chi2 = t([0.4, -0.3]), t([-0.2, 0.5])
    cq1, cq2 = t([1.0, 2.7]), t([1.0, 4.1])
    ham = TEOBResumSHamiltonian(
        nu, chi1, chi2, t([0.0, 380.0]), t([0.0, 420.0]),
        cq1=cq1, cq2=cq2)
    r = t([[6.0, 9.0], [7.0, 11.0]])
    a1 = chi1[:, None] * ham.x1[:, None]
    a2 = chi2[:, None] * ham.x2[:, None]
    a_q_sq = cq1[:, None] * a1.square() + 2.0 * a1 * a2 + cq2[:, None] * a2.square()
    expected = r.square() + a_q_sq * (1.0 + 2.0 / r)
    torch.testing.assert_close(ham.centrifugal_radius_squared(r), expected,
                               rtol=2e-15, atol=2e-15)


def test_centrifugal_radius_reduces_to_r_for_nonspinning_binaries():
    ham = TEOBResumSHamiltonian(t(.23), t(0.0), t(0.0), t(380.0), t(420.0))
    r = t([5.0, 8.0, 20.0])
    torch.testing.assert_close(ham.centrifugal_radius_squared(r), r.square(),
                               rtol=0.0, atol=0.0)


def test_spin_hamiltonian_uses_rc_in_both_orbital_momentum_terms():
    ham = TEOBResumSHamiltonian(
        t(.23), t(.35), t(-.15), t(0.0), t(0.0), cq1=t(1.0), cq2=t(1.0))
    r, pr, pphi = t(8.0), t(.07), t(3.1)
    rc2 = ham.centrifugal_radius_squared(r)
    a, _, _ = ham.metric_potentials(r)
    z3 = 2.0 * ham.nu * (4.0 - 3.0 * ham.nu)
    expected_even = torch.sqrt(pr.square() + a * (
        1.0 + pphi.square() / rc2 + z3 * pr.pow(4) / rc2))
    got = ham.effective_hamiltonian(r, pr, pphi)
    gplus, gminus = ham.gyro_factors(r)
    odd = pphi * (gplus * ham.a_plus + gminus * ham.delta * ham.a_minus) / (
        r.pow(3) + ham.a_plus.square() * (r + 2.0))
    torch.testing.assert_close(got, expected_even + odd, rtol=2e-15, atol=2e-15)


def test_metric_a_uses_the_published_uc_map():
    ham = TEOBResumSHamiltonian(
        t(.23), t(.25), t(-.1), t(0.0), t(0.0), cq1=t(1.0), cq2=t(1.0))
    r = t([6.0, 10.0])
    rc = torch.sqrt(ham.centrifugal_radius_squared(r))
    u, uc = 1.0 / r, 1.0 / rc
    a0c = ham.point_mass_metric_potentials(rc)[0]
    expected = (1.0 + 2.0 * uc) / (1.0 + 2.0 * u) * (
        a0c + ham.tidal_metric_potential(uc))
    torch.testing.assert_close(ham.metric_potentials(r)[0], expected,
                               rtol=2e-15, atol=2e-15)
