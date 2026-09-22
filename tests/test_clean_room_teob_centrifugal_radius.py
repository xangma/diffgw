"""Nagar et al., arXiv:1806.01772v2 Eqs. (24)--(25), independent algebra checks."""

import torch
from torchwave.approximants.teobresums import _teobresums_centrifugal_radius


def t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def test_centrifugal_radius_matches_published_quadratic_cq_map():
    nu = t([0.20, 0.24])
    chi1, chi2 = t([0.4, -0.3]), t([-0.2, 0.5])
    cq1, cq2 = t([1.0, 2.7]), t([1.0, 4.1])
    r = t([[6.0, 9.0], [7.0, 11.0]])
    delta = torch.sqrt(1 - 4 * nu)
    a1 = chi1[:, None] * (1 + delta[:, None]) / 2
    a2 = chi2[:, None] * (1 - delta[:, None]) / 2
    a_q_sq = cq1[:, None] * a1.square() + 2 * a1 * a2 + cq2[:, None] * a2.square()
    expected = r.square() + a_q_sq * (1 + 2 / r)
    actual = _teobresums_centrifugal_radius(r, nu, chi1, chi2, cq1, cq2).square()
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)


def test_centrifugal_radius_reduces_to_r_for_nonspinning_binaries():
    r = t([5., 8., 20.])
    actual = _teobresums_centrifugal_radius(r, t(.23), t(0.), t(0.))
    torch.testing.assert_close(actual, r, rtol=0., atol=0.)
