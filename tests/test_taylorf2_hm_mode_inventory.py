"""Clean-room leading-PN inventory checks for TaylorF2_HM.

The (3,1) and (4,3) equations are the explicit Fourier-domain leading terms
in García-Quirós et al., Phys. Rev. D 102, 064002 (2020), Appendix FPN,
Eqs. (31amp) and (43amp).  The (4,2) equation is the explicit leading SPA
term in Mezzasoma & Yunes, Phys. Rev. D 106, 024026 (2022), Eq. (2.21g).
"""

import math

import pytest
import torch

from torchwave.approximants.taylorf2_hm import (
    TaylorF2_HM,
    _swsh_projection_coefficients,
)


def test_published_leading_ratios_31_33_and_43():
    """Leading terms have exact ratios after the common SPA normalization."""
    f = torch.tensor([40.0, 80.0, 160.0], dtype=torch.float64)
    m1, m2 = 30.0, 10.0
    eta = m1 * m2 / (m1 + m2) ** 2
    v22 = (math.pi * (m1 + m2) * 4.925490947641267e-6 * f).pow(1.0 / 3.0)
    inclination = torch.tensor(0.7, dtype=torch.float64)
    phi0 = torch.tensor(0.13, dtype=torch.float64)

    _, _, modes = TaylorF2_HM(modes=((2, 1), (2, 2), (3, 1), (3, 3), (4, 2), (4, 3)))(
        f,
        m1,
        m2,
        inclination=inclination,
        phi0=phi0,
        f_isco_cutoff=False,
        return_modes=True,
    )

    d21, _ = _swsh_projection_coefficients(2, 1, inclination, phi0)
    d22, _ = _swsh_projection_coefficients(2, 2, inclination, phi0)
    d31, _ = _swsh_projection_coefficients(3, 1, inclination, phi0)
    d33, _ = _swsh_projection_coefficients(3, 3, inclination, phi0)
    d42, _ = _swsh_projection_coefficients(4, 2, inclination, phi0)
    d43, _ = _swsh_projection_coefficients(4, 3, inclination, phi0)

    h21 = modes[(2, 1)][0] / d21
    h22 = modes[(2, 2)][0] / d22
    h31 = modes[(3, 1)][0] / d31
    h33 = modes[(3, 3)][0] / d33
    h42 = modes[(4, 2)][0] / d42
    h43 = modes[(4, 3)][0] / d43

    # H31/H21 = [delta*v22/(12*sqrt(7)*2^(5/6))]
    #            / [delta*v22/(3*2^(1/3))] = 1/(4*sqrt(14)).
    ratio_31 = h31 / h21
    assert torch.allclose(
        ratio_31.real,
        torch.full_like(v22, 1.0 / (4.0 * math.sqrt(14.0))),
        rtol=2e-12,
        atol=1e-14,
    )
    assert torch.max(torch.abs(ratio_31.imag)).item() < 2e-13

    # H33/H22 = (3/4)*sqrt(5/7)*(3/2)^(5/6)*delta*v22.
    expected_33 = (
        (3.0 / 4.0)
        * math.sqrt(5.0 / 7.0)
        * (3.0 / 2.0) ** (5.0 / 6.0)
        * ((m1 - m2) / (m1 + m2))
        * v22
    )
    ratio_33 = torch.abs(h33 / h22)
    assert torch.allclose(ratio_33, expected_33, rtol=2e-12, atol=1e-14)

    # H42/H22 = sqrt(5)/63 * (1 - 3*eta) * v22^2.
    expected_42 = (math.sqrt(5.0) / 63.0) * (1.0 - 3.0 * eta) * v22.pow(2)
    ratio_42 = h42 / h22
    # Eq. (2.21g) is real and positive in this convention; checking the
    # complex ratio also locks the m=2 phase mapping, not only its magnitude.
    assert torch.allclose(ratio_42.real, expected_42, rtol=2e-12, atol=1e-14)
    assert torch.max(torch.abs(ratio_42.imag)).item() < 2e-13

    # H43/H22 = (3/4)*sqrt(3/35)*(3/2)^(1/6)*delta*(1-2*eta)*v22^3.
    expected_43 = (
        (3.0 / 4.0)
        * math.sqrt(3.0 / 35.0)
        * (3.0 / 2.0) ** (1.0 / 6.0)
        * ((m1 - m2) / (m1 + m2))
        * (1.0 - 2.0 * eta)
        * v22.pow(3)
    )
    ratio_43 = torch.abs(h43 / h22)
    assert torch.allclose(ratio_43, expected_43, rtol=2e-12, atol=1e-14)


def test_new_mode_projections_match_standard_spin_weighted_harmonics():
    """The 31/42/43 projection polynomials use the standard s=-2 basis."""
    iota = torch.tensor(0.73, dtype=torch.float64)
    phi0 = torch.tensor(0.21, dtype=torch.float64)
    c = torch.cos(iota)
    s = torch.sin(iota)
    op = 1.0 + c
    om = 1.0 - c
    ep = torch.complex(torch.cos(phi0), torch.sin(phi0))
    em = torch.conj(ep)

    n31 = math.sqrt(70.0 / (256.0 * math.pi))
    y31 = n31 * s * op * (3.0 * c - 1.0) * ep
    ym31 = n31 * s * om * (1.0 + 3.0 * c) * em
    expected_d31 = 0.5 * (ym31 - torch.conj(y31))
    d31, _ = _swsh_projection_coefficients(3, 1, iota, phi0)
    assert torch.allclose(d31, expected_d31, rtol=1e-14, atol=1e-15)

    n42 = 3.0 / (8.0 * math.sqrt(math.pi))
    y42 = n42 * op.square() * (7.0 * c.square() - 7.0 * c + 1.0) * torch.exp(2j * phi0)
    ym42 = n42 * om.square() * (7.0 * c.square() + 7.0 * c + 1.0) * torch.exp(-2j * phi0)
    expected_d42 = 0.5 * (ym42 + torch.conj(y42))
    expected_x42 = 0.5j * (ym42 - torch.conj(y42))
    d42, x42 = _swsh_projection_coefficients(4, 2, iota, phi0)
    assert torch.allclose(d42, expected_d42, rtol=1e-14, atol=1e-15)
    assert torch.allclose(x42, expected_x42, rtol=1e-14, atol=1e-15)

    n43 = 3.0 * math.sqrt(14.0 / (256.0 * math.pi))
    y43 = n43 * s * op.square() * (1.0 - 2.0 * c) * torch.exp(3j * phi0)
    ym43 = n43 * s * om.square() * (1.0 + 2.0 * c) * torch.exp(-3j * phi0)
    expected_d43 = 0.5 * (ym43 + torch.conj(y43))
    d43, _ = _swsh_projection_coefficients(4, 3, iota, phi0)
    assert torch.allclose(d43, expected_d43, rtol=1e-14, atol=1e-15)


def test_new_modes_equal_mass_zero_and_42_is_supported():
    f = torch.tensor([40.0, 80.0], dtype=torch.float64)
    _, _, modes = TaylorF2_HM()(f, 10.0, 10.0, f_isco_cutoff=False, return_modes=True)
    for mode in ((3, 1), (4, 3)):
        assert torch.all(modes[mode][0] == 0.0)
        assert torch.all(modes[mode][1] == 0.0)

    _, _, modes = TaylorF2_HM(modes=((4, 2),))(f, 10.0, 10.0, f_isco_cutoff=False, return_modes=True)
    assert torch.all(torch.isfinite(modes[(4, 2)][0]))
    assert torch.any(torch.abs(modes[(4, 2)][0]) > 0.0)


def test_42_mode_is_differentiable_in_intrinsic_and_orientation_parameters():
    """The new amplitude and SWSH projection preserve autograd gradients."""
    f = torch.tensor([40.0, 80.0], dtype=torch.float64)
    m1 = torch.tensor(28.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(11.0, dtype=torch.float64, requires_grad=True)
    inclination = torch.tensor(0.61, dtype=torch.float64, requires_grad=True)
    phi0 = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)

    _, _, modes = TaylorF2_HM(modes=((4, 2),))(
        f,
        m1,
        m2,
        inclination=inclination,
        phi0=phi0,
        f_isco_cutoff=False,
        return_modes=True,
    )
    h_plus, h_cross = modes[(4, 2)]
    loss = (h_plus.abs().square() + h_cross.abs().square()).sum()
    grads = torch.autograd.grad(loss, (m1, m2, inclination, phi0))
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_mode_sequences_accept_json_lists():
    """Frozen JSON inventories deserialize mode tuples as lists."""
    f = torch.tensor([40.0, 80.0], dtype=torch.float64)
    expected = TaylorF2_HM(modes=((2, 2),))(f, 10.0, 10.0, f_isco_cutoff=False)
    constructed = TaylorF2_HM(modes=[[2, 2]])(
        f, 10.0, 10.0, f_isco_cutoff=False
    )
    per_call = TaylorF2_HM()(f, 10.0, 10.0, modes=[[2, 2]], f_isco_cutoff=False)
    for actual_pair in (constructed, per_call):
        for actual, reference in zip(actual_pair, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_new_modes_batch_float32_device(device):
    device = torch.device(device)
    f = torch.linspace(40.0, 120.0, 6, dtype=torch.float32, device=device)
    m1 = torch.tensor([20.0, 30.0], dtype=torch.float32, device=device)
    m2 = torch.tensor([10.0, 12.0], dtype=torch.float32, device=device)
    inc = torch.tensor([0.4, 0.8], dtype=torch.float32, device=device)
    hp, hc, modes = TaylorF2_HM(modes=((3, 1), (4, 3)))(
        f,
        m1,
        m2,
        inclination=inc,
        f_isco_cutoff=False,
        return_modes=True,
    )
    assert hp.shape == (2, 6)
    assert hc.shape == (2, 6)
    assert hp.dtype == torch.complex64
    assert all(value[0].device == device for value in modes.values())
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
