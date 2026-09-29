"""Independent Clean-Room Acceptance Tests for PhenomHM Mapping Reconstruction.

Strict Clean-Room Literature Implementation:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Mishra et al. (2016), Phys. Rev. D 94, 084053 (arXiv:1609.00727)
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
"""

import math
import pytest
import torch

from torchwave.constants import MSUN_SEC, PI
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM,
    frequency_mapping_amplitude,
    frequency_mapping_phase,
    h_hat_multipole,
    _swsh_projection_coefficients,
    QNM_FITS,
)


def test_qnm_fits_and_supported_modes():
    """Verify all 6 modes are in QNM_FITS and IMRPhenomHM.SUPPORTED_MODES."""
    expected_modes = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
    assert IMRPhenomHM.SUPPORTED_MODES == expected_modes
    assert len(IMRPhenomHM.SUPPORTED_MODES) == 6
    for lm in expected_modes:
        assert lm in QNM_FITS, f"Mode {lm} missing from QNM_FITS"
        assert "f" in QNM_FITS[lm] and "q" in QNM_FITS[lm]
        assert len(QNM_FITS[lm]["f"]) == 3
        assert len(QNM_FITS[lm]["q"]) == 3

    # Berti et al. (2006) Table VIII check for (4, 3)
    f43 = QNM_FITS[(4, 3)]["f"]
    q43 = QNM_FITS[(4, 3)]["q"]
    assert math.isclose(f43[0], 1.6869, rel_tol=1e-5)
    assert math.isclose(f43[1], -0.8862, rel_tol=1e-5)
    assert math.isclose(f43[2], 0.2822, rel_tol=1e-5)
    assert math.isclose(q43[0], 1.4812, rel_tol=1e-5)
    assert math.isclose(q43[1], 2.8096, rel_tol=1e-5)
    assert math.isclose(q43[2], -0.4271, rel_tol=1e-5)


def test_restricted_pn_m_minus_one_modes_follow_published_spin_terms():
    """Check the restricted 1.5PN H21/H32/H43 equations independently.

    London et al. prescribe the 1.5PN spin-dependent truncation for the
    |m|=l-1 modes; the coefficients below are the corresponding restricted
    expressions in Mishra et al., Eqs. (12b), (12d), and (12g).
    """
    eta = torch.tensor(0.21, dtype=torch.float64)
    delta = torch.tensor(0.37, dtype=torch.float64)
    chi_s = torch.tensor(-0.18, dtype=torch.float64)
    chi_a = torch.tensor(0.43, dtype=torch.float64)
    v = torch.tensor(0.17, dtype=torch.float64)

    h21 = h_hat_multipole(2, 1, v, delta, eta, chi_s, chi_a)
    h32 = h_hat_multipole(3, 2, v, delta, eta, chi_s, chi_a)
    h43 = h_hat_multipole(4, 3, v, delta, eta, chi_s, chi_a)
    expected21 = -math.sqrt(2.0) / 3.0 * (
        delta * v - 1.5 * (chi_a + delta * chi_s) * v**2
    )
    expected32 = -math.sqrt(5.0 / 7.0) / 3.0 * (
        (1.0 - 3.0 * eta) * v**2 + 4.0 * eta * chi_s * v**3
    )
    expected43 = -0.75 * math.sqrt(3.0 / 35.0) * (
        delta * (1.0 - 2.0 * eta) * v**3
        + 2.5 * eta * (chi_a - delta * chi_s) * v**4
    )
    torch.testing.assert_close(h21, expected21, rtol=0.0, atol=0.0)
    torch.testing.assert_close(h32, expected32, rtol=0.0, atol=0.0)
    torch.testing.assert_close(h43, expected43, rtol=0.0, atol=0.0)


def test_frequency_mappings_continuity():
    """Verify continuity of piecewise frequency mappings f_22^A and f_22^phi across f0 and f_RD."""
    model = IMRPhenomHM()
    m1 = torch.tensor(35.0, dtype=torch.float64)
    m2 = torch.tensor(20.0, dtype=torch.float64)
    s1 = torch.tensor(0.2, dtype=torch.float64)
    s2 = torch.tensor(-0.1, dtype=torch.float64)
    m_sec = (m1 + m2) * MSUN_SEC

    f_rd_22_geom, _ = model.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
    f_rd_22 = f_rd_22_geom / m_sec

    eps = torch.tensor(1e-7, dtype=torch.float64)

    for l, m in IMRPhenomHM.SUPPORTED_MODES:
        if (l, m) == (2, 2):
            continue
        f_rd_lm_geom, _ = model.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
        f_rd_lm = f_rd_lm_geom / m_sec

        f0_amp = (0.018 / m_sec) * (f_rd_lm / f_rd_22)
        # Exercise London's published higher-mode mapping constants directly;
        # the runtime carrier convention is audited separately.
        f0_phi = (0.014 / m_sec) * (f_rd_lm / f_rd_22)

        # 1. Check f_22^A continuity across f0_amp
        fa_left = frequency_mapping_amplitude(f0_amp - eps, f0_amp, f_rd_lm, f_rd_22, m)
        fa_mid = frequency_mapping_amplitude(f0_amp, f0_amp, f_rd_lm, f_rd_22, m)
        fa_right = frequency_mapping_amplitude(f0_amp + eps, f0_amp, f_rd_lm, f_rd_22, m)
        assert torch.allclose(fa_left, fa_mid, atol=1e-6)
        assert torch.allclose(fa_right, fa_mid, atol=1e-6)

        # 2. Check f_22^A continuity across f_rd_lm
        fa_rd_left = frequency_mapping_amplitude(f_rd_lm - eps, f0_amp, f_rd_lm, f_rd_22, m)
        fa_rd_mid = frequency_mapping_amplitude(f_rd_lm, f0_amp, f_rd_lm, f_rd_22, m)
        fa_rd_right = frequency_mapping_amplitude(f_rd_lm + eps, f0_amp, f_rd_lm, f_rd_22, m)
        assert torch.allclose(fa_rd_left, fa_rd_mid, atol=1e-6)
        assert torch.allclose(fa_rd_right, fa_rd_mid, atol=1e-6)
        assert torch.allclose(fa_rd_mid, f_rd_22, atol=1e-6)

        # 3. Check f_22^phi continuity across f0_phi
        fp_left = frequency_mapping_phase(f0_phi - eps, f0_phi, f_rd_lm, f_rd_22, m)
        fp_mid = frequency_mapping_phase(f0_phi, f0_phi, f_rd_lm, f_rd_22, m)
        fp_right = frequency_mapping_phase(f0_phi + eps, f0_phi, f_rd_lm, f_rd_22, m)
        assert torch.allclose(fp_left, fp_mid, atol=1e-6)
        assert torch.allclose(fp_right, fp_mid, atol=1e-6)

        # 4. Check f_22^phi continuity across f_rd_lm
        fp_rd_left = frequency_mapping_phase(f_rd_lm - eps, f0_phi, f_rd_lm, f_rd_22, m)
        fp_rd_mid = frequency_mapping_phase(f_rd_lm, f0_phi, f_rd_lm, f_rd_22, m)
        fp_rd_right = frequency_mapping_phase(f_rd_lm + eps, f0_phi, f_rd_lm, f_rd_22, m)
        assert torch.allclose(fp_rd_left, fp_rd_mid, atol=1e-6)
        assert torch.allclose(fp_rd_right, fp_rd_mid, atol=1e-6)
        assert torch.allclose(fp_rd_mid, f_rd_22, atol=1e-6)


def test_phase_continuity_across_boundaries():
    """Verify continuity of phi_lm(f) at the runtime transition and ringdown joins."""
    model = IMRPhenomHM()
    m1 = torch.tensor(35.0, dtype=torch.float64)
    m2 = torch.tensor(20.0, dtype=torch.float64)
    s1 = torch.tensor(0.2, dtype=torch.float64)
    s2 = torch.tensor(-0.1, dtype=torch.float64)
    m_sec = (m1 + m2) * MSUN_SEC

    f_rd_22_geom, _ = model.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
    f_rd_22 = f_rd_22_geom / m_sec
    eps = torch.tensor(1e-6, dtype=torch.float64)

    for l, m in IMRPhenomHM.SUPPORTED_MODES:
        if (l, m) == (2, 2):
            continue
        f_rd_lm_geom, _ = model.mode_qnm_frequencies(l, m, m1, m2, s1, s2)
        f_rd_lm = f_rd_lm_geom / m_sec
        # Match IMRPhenomHM's PhenomD carrier phase join, Mf=0.018.  This is
        # distinct from London's separately stated higher-mode f_0^phi value.
        f0_phi = (0.018 / m_sec) * (f_rd_lm / f_rd_22)

        # Sample tight window around f0_phi
        f_grid_f0 = torch.tensor([f0_phi - eps, f0_phi, f0_phi + eps], dtype=torch.float64)
        _, _, modes_f0 = model(
            f=f_grid_f0,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            modes=[(l, m)],
            return_unprojected_modes=True,
        )
        h_f0 = modes_f0[(l, m)]
        # Phase jump across boundary is angle(h(f+eps) * conj(h(f-eps)))
        dphi_f0 = torch.angle(h_f0[2] * torch.conj(h_f0[0]))
        assert torch.abs(dphi_f0).item() < 1e-4, f"Phase jump at f0 for mode ({l}, {m}): {dphi_f0.item():.3e}"

        # Sample tight window around f_rd_lm
        f_grid_rd = torch.tensor([f_rd_lm - eps, f_rd_lm, f_rd_lm + eps], dtype=torch.float64)
        _, _, modes_rd = model(
            f=f_grid_rd,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            modes=[(l, m)],
            return_unprojected_modes=True,
        )
        h_rd = modes_rd[(l, m)]
        dphi_rd = torch.angle(h_rd[2] * torch.conj(h_rd[0]))
        assert torch.abs(dphi_rd).item() < 1e-4, f"Phase jump at f_rd for mode ({l}, {m}): {dphi_rd.item():.3e}"


def test_phase_derivative_relation():
    """Verify the mapped phase derivative, including London's Eq. (14)."""
    model = IMRPhenomHM()
    m1 = torch.tensor(35.0, dtype=torch.float64)
    m2 = torch.tensor(20.0, dtype=torch.float64)
    s1 = torch.tensor(0.2, dtype=torch.float64)
    s2 = torch.tensor(-0.1, dtype=torch.float64)
    m_sec = (m1 + m2) * MSUN_SEC

    f_rd_22_geom, f_damp_22 = model.mode_qnm_frequencies(2, 2, m1, m2, s1, s2)
    f_rd_22 = f_rd_22_geom / m_sec
    f_rd_33_geom, f_damp_33 = model.mode_qnm_frequencies(3, 3, m1, m2, s1, s2)
    f_rd_33 = f_rd_33_geom / m_sec
    # Match the PhenomD carrier phase join used by this runtime (Mf=0.018).
    # London et al.'s HM mapping paper separately states f_0^phi=0.014;
    # see docs/reviews/2026-09-12-clean-room/pass-30-phenomhm-transition-audit.md.
    f0_phi = (0.018 / m_sec) * (f_rd_33 / f_rd_22)

    # Test point inside Region 2 (intermediate regime)
    f_mid = 0.5 * (f0_phi + f_rd_33)
    df = torch.tensor(1e-4, dtype=torch.float64)

    s2_slope = (f_rd_22 - (2.0 / 3.0) * f0_phi) / (f_rd_33 - f0_phi)
    kappa_2 = 1.0 / s2_slope

    # Mode (3, 3) phase derivative via phase difference of h33
    f_pts = torch.tensor([f_mid - df, f_mid + df], dtype=torch.float64)
    _, _, modes = model(f=f_pts, mass1=m1, mass2=m2, chi1=s1, chi2=s2, modes=[(3, 3)], return_unprojected_modes=True)
    h33 = modes[(3, 3)]
    # h = A * exp(-i * phi) -> angle(h1 * conj(h0)) = -(phi1 - phi0)
    dphi33 = -torch.angle(h33[1] * torch.conj(h33[0]))
    dphi33_df = dphi33 / (2.0 * df)

    # Base (2, 2) phase evaluated directly at mapped frequencies
    f22_mid = s2_slope * (f_mid - f0_phi) + (2.0 / 3.0) * f0_phi
    f22_pts = torch.tensor([f22_mid - df, f22_mid + df], dtype=torch.float64)
    _, _, phi22_pts = model.base_22(f=f22_pts, mass1=m1, mass2=m2, chi1=s1, chi2=s2, phic=0.0, tc=0.0, return_phase=True)
    dphi22_df22 = (phi22_pts[1] - phi22_pts[0]) / (2.0 * df)

    # London Eq. (14) changes the arctangent derivative, and C1 matching subtracts
    # that derivative change at D's merger-ringdown join.
    eta = m1 * m2 / (m1 + m2).square()
    chi_pn = 0.5 * (s1 + s2) * (1.0 - 76.0 * eta / 113.0) + (m1 - m2) / (m1 + m2) * 0.5 * (s1 - s2)
    a4 = model.base_22.eval_table_v("alpha4", eta, chi_pn)
    a5 = model.base_22.eval_table_v("alpha5", eta, chi_pn)
    ratio = f_rd_22_geom / f_rd_33_geom
    width = ratio * f_damp_33

    def lorentzian_change(mf):
        shift = mf - a5 * f_rd_22_geom
        return a4 / eta * (
            ratio * width / (width.square() + shift.square())
            - f_damp_22 / (f_damp_22.square() + shift.square())
        )

    dphi22_df22 += m_sec * (
        lorentzian_change(f22_mid * m_sec) - lorentzian_change(0.5 * f_rd_22_geom)
    )

    rel_err = torch.abs(dphi33_df - dphi22_df22) / torch.abs(dphi22_df22)
    assert rel_err.item() < 1e-4, f"Phase derivative relation error {rel_err.item():.4e} exceeds 1e-4"


def test_all_six_modes_finite_strains():
    """Verify all 6 modes produce finite strains and non-zero unprojected modes."""
    model = IMRPhenomHM()
    f = torch.linspace(25.0, 600.0, 100, dtype=torch.float64)

    hp, hc, mode_dict = model(
        f=f,
        mass1=40.0,
        mass2=15.0,
        chi1=0.25,
        chi2=-0.15,
        inclination=math.pi / 3.0,
        return_modes=True,
    )

    assert torch.all(torch.isfinite(hp.real)) and torch.all(torch.isfinite(hp.imag))
    assert torch.all(torch.isfinite(hc.real)) and torch.all(torch.isfinite(hc.imag))
    assert len(mode_dict) == 6

    for lm in IMRPhenomHM.SUPPORTED_MODES:
        assert lm in mode_dict, f"Mode {lm} missing from mode_dict"
        hp_m, hc_m = mode_dict[lm]
        assert torch.all(torch.isfinite(hp_m.real)) and torch.all(torch.isfinite(hp_m.imag))
        assert torch.all(torch.isfinite(hc_m.real)) and torch.all(torch.isfinite(hc_m.imag))
        assert torch.any(hp_m.abs() > 0.0), f"Mode {lm} h_plus is identically zero"
        assert torch.any(hc_m.abs() > 0.0), f"Mode {lm} h_cross is identically zero"


def test_body_exchange_parity():
    """Verify body exchange parity: m1 <-> m2, chi1 <-> chi2."""
    model = IMRPhenomHM()
    f = torch.linspace(25.0, 400.0, 80, dtype=torch.float64)
    m1, m2 = 45.0, 18.0
    c1, c2 = 0.35, -0.25

    # Configuration A: (m1, m2, c1, c2)
    _, _, modes_a = model(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        return_unprojected_modes=True,
    )

    # Configuration B: (m2, m1, c2, c1)
    _, _, modes_b = model(
        f=f,
        mass1=m2,
        mass2=m1,
        chi1=c2,
        chi2=c1,
        return_unprojected_modes=True,
    )

    for l, m in IMRPhenomHM.SUPPORTED_MODES:
        ha = modes_a[(l, m)]
        hb = modes_b[(l, m)]

        # 1. Mode magnitudes must be strictly invariant under body exchange
        mag_a = ha.abs()
        mag_b = hb.abs()
        rel_diff = torch.linalg.norm(mag_a - mag_b) / (torch.linalg.norm(mag_a) + 1e-30)
        assert rel_diff < 1e-12, f"Mode ({l}, {m}) magnitude differs under body exchange: {rel_diff:.3e}"

        # 2. Mode parity under body exchange: (-1)^m
        expected_sign = (-1.0) ** m
        parity_diff = torch.linalg.norm(hb - expected_sign * ha) / (torch.linalg.norm(ha) + 1e-30)
        assert parity_diff < 1e-12, f"Mode ({l}, {m}) parity error under body exchange: {parity_diff:.3e}"

    # 3. Full waveform with rotated azimuth phi0 -> phi0 + pi
    hp_a, hc_a = model(f=f, mass1=m1, mass2=m2, chi1=c1, chi2=c2, inclination=math.pi / 4.0, phi0=0.0)
    hp_b, hc_b = model(f=f, mass1=m2, mass2=m1, chi1=c2, chi2=c1, inclination=math.pi / 4.0, phi0=math.pi)
    assert torch.allclose(hp_a, hp_b, rtol=1e-10, atol=1e-28)
    assert torch.allclose(hc_a, hc_b, rtol=1e-10, atol=1e-28)


def test_equal_mass_unequal_spin_odd_mode_contracts():
    """Check the production London odd-mode contract and the public PN helper."""
    model = IMRPhenomHM()
    f = torch.linspace(25.0, 300.0, 80, dtype=torch.float64)

    # 1. Equal-mass, equal-spin: odd modes MUST vanish identically
    _, _, modes_eq = model(
        f=f,
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=0.2,
        return_unprojected_modes=True,
    )
    for lm in [(2, 1), (3, 3), (4, 3)]:
        assert torch.allclose(modes_eq[lm].abs(), torch.zeros_like(modes_eq[lm].abs()), atol=1e-28), (
            f"Mode {lm} should vanish for equal-mass equal-spin"
        )

    # 2. Equal-mass, unequal-spin: the production London (2, 1) mode retains
    # its displayed spin term.  The published production truncation keeps the
    # (4, 3) leading antisymmetric-mass term, so it vanishes at equal masses;
    # the public full PN helper still exposes its higher-order spin term.
    _, _, modes_spin = model(
        f=f,
        mass1=30.0,
        mass2=30.0,
        chi1=0.5,
        chi2=-0.5,
        return_unprojected_modes=True,
    )
    h21 = modes_spin[(2, 1)]
    h43 = modes_spin[(4, 3)]
    assert torch.any(h21.abs() > 0.0), "Mode (2, 1) should not vanish when chi1 != chi2"
    assert torch.allclose(h43, torch.zeros_like(h43), atol=1e-28)

    from torchwave.approximants.imrphenom_hm import h_hat_multipole

    v = torch.full_like(f, 0.2)
    h43_full_pn = h_hat_multipole(
        4,
        3,
        v,
        torch.zeros_like(v),
        torch.full_like(v, 0.25),
        torch.zeros_like(v),
        torch.full_like(v, 0.5),
    )
    assert torch.any(h43_full_pn.abs() > 0.0)
    assert torch.max(h21.abs()).item() > 1e-26, "Mode (2, 1) amplitude unexpectedly small"


def test_autograd_physical_parameters():
    """Verify autograd differentiability w.r.t. masses, spins, distance, inclination, and phic."""
    model = IMRPhenomHM()
    f = torch.linspace(30.0, 250.0, 40, dtype=torch.float64)

    m1 = torch.tensor(38.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(22.0, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(-0.15, dtype=torch.float64, requires_grad=True)
    dist = torch.tensor(150.0, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(math.pi / 3.0, dtype=torch.float64, requires_grad=True)
    phi_c = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    t_c = torch.tensor(0.002, dtype=torch.float64, requires_grad=True)

    hp, hc = model(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        distance=dist,
        inclination=inc,
        phic=phi_c,
        tc=t_c,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    params = [
        (m1, "mass1"),
        (m2, "mass2"),
        (c1, "chi1"),
        (c2, "chi2"),
        (dist, "distance"),
        (inc, "inclination"),
        (phi_c, "phic"),
        (t_c, "tc"),
    ]

    for p, name in params:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is non-finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"


def test_batched_evaluation():
    """Verify batched evaluation matches individual unbatched evaluations."""
    model = IMRPhenomHM()
    f = torch.linspace(25.0, 300.0, 60, dtype=torch.float64)
    m1_b = torch.tensor([32.0, 42.0, 26.0], dtype=torch.float64)
    m2_b = torch.tensor([22.0, 18.0, 16.0], dtype=torch.float64)
    c1_b = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
    c2_b = torch.tensor([-0.1, 0.2, 0.0], dtype=torch.float64)

    hp_b, hc_b = model(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=c1_b,
        chi2=c2_b,
        inclination=math.pi / 4.0,
    )
    assert hp_b.shape == (3, len(f))

    for i in range(3):
        hp_s, hc_s = model(
            f=f,
            mass1=m1_b[i],
            mass2=m2_b[i],
            chi1=c1_b[i],
            chi2=c2_b[i],
            inclination=math.pi / 4.0,
        )
        assert torch.allclose(hp_b[i], hp_s, rtol=1e-8, atol=1e-28)
        assert torch.allclose(hc_b[i], hc_s, rtol=1e-8, atol=1e-28)


def test_scalar_masses_broadcast_over_batched_nonmass_parameters():
    """Broadcast a shared mass pair across every batched physical parameter."""
    model = IMRPhenomHM()
    f = torch.linspace(30.0, 260.0, 36, dtype=torch.float64)
    chi1 = torch.tensor([0.22, -0.31], dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor([-0.11, 0.17], dtype=torch.float64, requires_grad=True)
    distance = torch.tensor([120.0, 190.0], dtype=torch.float64, requires_grad=True)
    inclination = torch.tensor([0.4, 1.1], dtype=torch.float64, requires_grad=True)
    phic = torch.tensor([0.15, -0.35], dtype=torch.float64, requires_grad=True)
    f_ref = torch.tensor([35.0, 55.0], dtype=torch.float64, requires_grad=True)

    hp_batch, hc_batch = model(
        f=f,
        mass1=35.0,
        mass2=21.0,
        chi1=chi1,
        chi2=chi2,
        distance=distance,
        inclination=inclination,
        phic=phic,
        f_ref=f_ref,
    )
    assert hp_batch.shape == (2, f.numel())
    assert hc_batch.shape == hp_batch.shape
    assert torch.isfinite(hp_batch.real).all() and torch.isfinite(hp_batch.imag).all()
    assert torch.isfinite(hc_batch.real).all() and torch.isfinite(hc_batch.imag).all()

    for i in range(2):
        hp_row, hc_row = model(
            f=f,
            mass1=35.0,
            mass2=21.0,
            chi1=chi1[i],
            chi2=chi2[i],
            distance=distance[i],
            inclination=inclination[i],
            phic=phic[i],
            f_ref=f_ref[i],
        )
        assert torch.allclose(hp_batch[i], hp_row, rtol=1e-9, atol=1e-28)
        assert torch.allclose(hc_batch[i], hc_row, rtol=1e-9, atol=1e-28)

    # A phase-sensitive loss exercises the broadcasted graph for all inputs.
    loss = (
        0.37 * hp_batch.real
        + 0.19 * hp_batch.imag
        - 0.23 * hc_batch.real
        + 0.11 * hc_batch.imag
    ).sum()
    loss.backward()
    for value, name in (
        (chi1, "chi1"),
        (chi2, "chi2"),
        (distance, "distance"),
        (inclination, "inclination"),
        (phic, "phic"),
        (f_ref, "f_ref"),
    ):
        assert value.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(value.grad).all(), f"Gradient for {name} is non-finite"
        assert torch.any(value.grad != 0.0), f"Gradient for {name} is zero"
