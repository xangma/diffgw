"""Archived legacy surrogate phasing check; not a release qualification test."""

import torch
from torchwave.approximants.rom.surrogate import SEOBNR_ROM


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


