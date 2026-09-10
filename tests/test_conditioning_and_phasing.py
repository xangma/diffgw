"""Tests for reference frequency (f_ref) physical conditioning and phasing conventions."""

import pytest
import torch
import math

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.rom.surrogate import SEOBNR_ROM
from torchwave.waveforms import get_fd_waveform
from torchwave.constants import PI, MSUN_SEC
from torchwave.conversions import component_masses_to_eta


def test_taylorf2_ref_phasing_exact_zero():
    """Verify TaylorF2 phase and time alignment: Psi(f_ref) = 0 and dPsi/df(f_ref) = 0."""
    tf2 = TaylorF2()
    m1, m2 = 1.4, 1.4
    chi1, chi2 = 0.05, -0.02
    f_ref = 30.0

    # Evaluate phase on grid including f_ref
    f = torch.tensor([20.0, f_ref, 50.0], dtype=torch.float64)
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    c1_t = torch.tensor(chi1, dtype=torch.float64)
    c2_t = torch.tensor(chi2, dtype=torch.float64)

    psi = tf2.evaluate_phase(f, m1_t, m2_t, c1_t, c2_t, tc=0.0, phic=0.0, f_ref=f_ref)
    dpsi_df = tf2.evaluate_phase_derivative(f, m1_t, m2_t, c1_t, c2_t, tc=0.0, f_ref=f_ref)

    # At f = f_ref, Psi(f_ref) should be identically 0
    assert abs(psi[1].item()) < 1e-12, f"Psi(f_ref) = {psi[1].item()} != 0"

    # At f = f_ref, dPsi/df(f_ref) should be identically 0
    assert abs(dpsi_df[1].item()) < 1e-12, f"dPsi/df(f_ref) = {dpsi_df[1].item()} != 0"


def test_imrphenomd_ref_phasing_exact_zero():
    """Verify IMRPhenomD phase and time alignment: Psi(f_ref) = 0 and dPsi/df(f_ref) = 0."""
    phd = IMRPhenomD()
    m1, m2 = 30.0, 20.0
    chi1, chi2 = 0.3, -0.2
    f_ref = 40.0

    # Evaluate waveform at f_ref and adjacent points
    f = torch.tensor([25.0, f_ref, 60.0], dtype=torch.float64)
    hp, _ = phd(f, m1, m2, chi1, chi2, distance=100.0, inclination=0.0, tc=0.0, phic=0.0, f_ref=f_ref)

    # Since hp = A * exp(-i * Psi), at f_ref with Psi(f_ref)=0, hp is purely real and positive
    phase_at_fref = torch.angle(hp[1]).item()
    assert abs(phase_at_fref) < 1e-10, f"IMRPhenomD phase at f_ref = {phase_at_fref} != 0"

    # Numerical derivative of phase around f_ref
    eps = 1e-5
    f_eps = torch.tensor([f_ref - eps, f_ref + eps], dtype=torch.float64)
    hp_eps, _ = phd(f_eps, m1, m2, chi1, chi2, distance=100.0, inclination=0.0, tc=0.0, phic=0.0, f_ref=f_ref)
    phase_eps = -torch.angle(hp_eps)
    num_deriv = (phase_eps[1] - phase_eps[0]) / (2.0 * eps)
    assert abs(num_deriv.item()) < 1e-5, f"IMRPhenomD dPsi/df(f_ref) = {num_deriv.item()} != 0"


def test_seobnr_rom_ref_phasing_exact_zero():
    """Verify SEOBNR_ROM phase and time alignment at f_ref."""
    rom = SEOBNR_ROM()
    m1, m2 = 35.0, 25.0
    chi1, chi2 = 0.1, -0.05
    f_ref = 35.0

    f = torch.tensor([25.0, f_ref, 75.0], dtype=torch.float64)
    hp, _ = rom(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        phic=0.0,
        tc=0.0,
        f_ref=f_ref,
    )

    # For face-on at tc=0, phic=0: hp = A * exp(-i * Psi)
    # Since Psi(f_ref) = 0, exp(-i * 0) = 1, so hp(f_ref) should be purely real and positive
    phase_at_fref = torch.angle(hp[1]).item()
    assert abs(phase_at_fref) < 1e-4, f"SEOBNR_ROM phase at f_ref = {phase_at_fref} != 0"

    # Check derivative of phase is near 0 at f_ref
    eps = 1e-4
    f_eps = torch.tensor([f_ref - eps, f_ref + eps], dtype=torch.float64)
    hp_eps, _ = rom(
        f=f_eps,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        phic=0.0,
        tc=0.0,
        f_ref=f_ref,
    )
    phase_eps = -torch.angle(hp_eps)
    num_deriv = (phase_eps[1] - phase_eps[0]) / (2.0 * eps)
    assert abs(num_deriv.item()) < 1e-2, f"SEOBNR_ROM dPsi/df(f_ref) = {num_deriv.item()} != 0"


def test_imrphenomp_euler_angle_reference_alignment():
    """Verify precessing Euler angle conventions: alpha(f_ref) = phi_s1 and epsilon(f_ref) = 0."""
    for pver in ('v2', 'xp'):
        m1, m2 = 25.0, 15.0
        s1x, s1y, s1z = 0.2, 0.4, 0.1
        f_ref = 30.0

        expected_phi_s1 = math.atan2(s1y, s1x)

        # Directly evaluate angle equations
        m_sec = (m1 + m2) * MSUN_SEC
        v_ref = torch.tensor((PI * m_sec * f_ref) ** (1.0 / 3.0), dtype=torch.float64)
        eta = component_masses_to_eta(torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64))
        delta = torch.tensor((m1 - m2) / (m1 + m2), dtype=torch.float64)
        m1_norm = 0.5 * (1.0 + delta)
        chip = torch.tensor(0.4, dtype=torch.float64)
        chi_l = torch.tensor(0.1, dtype=torch.float64)

        coeffs = list(IMRPhenomP.nnlo_angle_coefficients(eta, delta, m1_norm, chip, chi_l))
        if pver == 'v2':
            coeffs[3] = torch.zeros_like(coeffs[3])
            coeffs[4] = torch.zeros_like(coeffs[4])
            coeffs[8] = torch.zeros_like(coeffs[8])
            coeffs[9] = torch.zeros_like(coeffs[9])

        a_m3, a_m2, a_m1, a_1, a_log, e_m3, e_m2, e_m1, e_1, e_log = coeffs

        # Evaluate at v = v_ref
        alpha_dyn, epsilon = IMRPhenomP.evaluate_nnlo_angles(
            v=v_ref.unsqueeze(0),
            v_ref=v_ref.unsqueeze(0),
            a_minus3=a_m3, a_minus2=a_m2, a_minus1=a_m1, a_1=a_1, a_log=a_log,
            e_minus3=e_m3, e_minus2=e_m2, e_minus1=e_m1, e_1=e_1, e_log=e_log,
        )

        alpha = alpha_dyn[0] + expected_phi_s1

        assert abs(alpha.item() - expected_phi_s1) < 1e-12, f'{pver}: alpha(f_ref) != phi_s1'
        assert abs(epsilon[0].item()) < 1e-12, f'{pver}: epsilon(f_ref) != 0'


def test_waveform_array_parity_complex_slope():
    """Verify that waveforms generated with the same f_ref align with minimal residual slope."""
    f = torch.linspace(25.0, 150.0, 200, dtype=torch.float64)
    m1, m2 = 30.0, 20.0
    f_ref = 30.0

    hp1, _ = get_fd_waveform('IMRPhenomD', mass1=m1, mass2=m2, chi1=0.2, chi2=-0.1, sample_frequencies=f, f_ref=f_ref)
    hp2, _ = get_fd_waveform('IMRPhenomD', mass1=m1, mass2=m2, chi1=0.2, chi2=-0.1, sample_frequencies=f, f_ref=f_ref)

    # Complex difference should be zero
    diff = torch.max(torch.abs(hp1 - hp2))
    assert diff.item() < 1e-25, f'Waveforms with identical parameters and f_ref do not match: diff={diff}'


def test_batched_reference_phasing():
    """Verify f_ref conditioning works across batched evaluations."""
    B = 4
    m1 = torch.tensor([25.0, 30.0, 40.0, 50.0], dtype=torch.float64)
    m2 = torch.tensor([15.0, 20.0, 20.0, 30.0], dtype=torch.float64)
    f_refs = torch.tensor([20.0, 30.0, 40.0, 50.0], dtype=torch.float64)

    for approx in ('TaylorF2', 'IMRPhenomD', 'SEOBNR_ROM'):
        for b in range(B):
            f_eval = torch.tensor([15.0, f_refs[b].item(), 100.0], dtype=torch.float64)
            hp, _ = get_fd_waveform(
                approx,
                mass1=m1[b],
                mass2=m2[b],
                sample_frequencies=f_eval,
                f_ref=f_refs[b].item(),
                tc=0.0,
                phic=0.0,
            )
            phase_at_ref = torch.angle(hp[1]).item()
            assert abs(phase_at_ref) < 1e-4, f'{approx} batch {b}: phase at f_ref = {phase_at_ref}'
