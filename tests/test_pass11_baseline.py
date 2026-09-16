"""Equation checks for the bounded pass-11 NSBH remnant correction."""

import hashlib

import pytest
import torch

from torchwave.approximants.imrphenom_nsbh import (
    ZAPPA_MASS_COEFFS,
    ZAPPA_SPIN_COEFFS,
    bbh_remnant_uib2016,
    zappa_remnant_properties,
)


@pytest.mark.parametrize("ns_spin", [0.0, -0.8])
def test_zappa_tidal_polynomials_use_the_black_hole_spin(ns_spin):
    """Zappa Eqs. (4)-(6) use a_BH, not a mass-weighted two-spin variable."""
    mass1 = torch.tensor(7.0, dtype=torch.float64)
    mass2 = torch.tensor(1.4, dtype=torch.float64)
    chi_bh = torch.tensor(0.35, dtype=torch.float64)
    chi_ns = torch.tensor(ns_spin, dtype=torch.float64)
    lambda2 = torch.tensor(1200.0, dtype=torch.float64)

    mf, chif = zappa_remnant_properties(mass1, mass2, chi_bh, chi_ns, lambda2)
    eta = mass1 * mass2 / (mass1 + mass2).square()
    mf_bbh_ratio, chif_bbh = bbh_remnant_uib2016(mass1, mass2, chi_bh, chi_ns)

    # Zappa et al. (2019), Eqs. (4)-(6) and Table I: p_kj is linear in
    # a_BH and quadratic in nu.  These are the full precision coefficients
    # in the author-released BHNSremnant.py asset.
    def factor(coeff):
        p1 = ((coeff[0] * chi_bh + coeff[1]) + (coeff[2] * chi_bh + coeff[3]) * eta) * eta
        p2 = ((coeff[4] * chi_bh + coeff[5]) + (coeff[6] * chi_bh + coeff[7]) * eta) * eta
        p3 = ((coeff[8] * chi_bh + coeff[9]) + (coeff[10] * chi_bh + coeff[11]) * eta) * eta
        return p1, p2, p3

    p1_m, p2_m, p3_m = factor(ZAPPA_MASS_COEFFS)
    expected_mf = (mass1 + mass2) * mf_bbh_ratio * (
        1.0 + p1_m * lambda2 + p2_m * lambda2.square()
    ) / (1.0 + p3_m.square() * lambda2).square()

    p1_s, p2_s, p3_s = factor(ZAPPA_SPIN_COEFFS)
    expected_chif = chif_bbh * (
        1.0 + p1_s * lambda2 + p2_s * lambda2.square()
    ) / (1.0 + p3_s.square() * lambda2).square()
    expected_chif = torch.clamp(expected_chif, -0.999, 0.999)

    torch.testing.assert_close(mf, expected_mf, rtol=3e-14, atol=0.0)
    torch.testing.assert_close(chif, expected_chif, rtol=3e-14, atol=0.0)


def test_zappa_printed_table_equation_remains_reproducible():
    """Keep the rounded coefficients as a direct check of the displayed table."""
    eta = torch.tensor(7.0 * 1.4 / 8.4**2, dtype=torch.float64)
    chi = torch.tensor(0.35, dtype=torch.float64)
    printed = (
        (-1.83e-3, 2.39e-3, 4.29e-3, 9.80e-3,
         2.34e-7, -8.28e-7, -1.64e-6, 8.08e-6,
         -2.01e-2, 1.32e-1, 6.51e-2, -1.43e-1),
        (-5.44e-3, 7.91e-3, 2.33e-2, 2.48e-2,
         -8.57e-7, -2.82e-6, 6.61e-6, 4.29e-5,
         -3.04e-2, 2.55e-1, 1.48e-1, -4.28e-1),
    )
    values = []
    for coeff in printed:
        values.extend(
            (((coeff[i] * chi + coeff[i + 1]) +
              (coeff[i + 2] * chi + coeff[i + 3]) * eta) * eta).item()
            for i in (0, 4, 8)
        )
    assert values == pytest.approx([
        4.609934413580246e-4, 4.1166666666666665e-8, 1.5037287808641974e-2,
        1.469872685185185e-3, 4.388476080246911e-7, 2.6681944444444442e-2,
    ], rel=0.0, abs=1e-18)


def test_zappa_full_precision_asset_hash():
    """Pin the coefficient transcription to the author asset provenance."""
    canonical = repr(ZAPPA_MASS_COEFFS + ZAPPA_SPIN_COEFFS).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        "378339f004dc13d5fd024ef6615ac1db5020140f141935a95444970eab4c6450"
    )


def test_zappa_lambda_zero_is_the_supplied_bbh_baseline():
    """The fit's Lambda -> 0 construction preserves both supplied BBH spins."""
    mass1 = torch.tensor(6.0, dtype=torch.float64)
    mass2 = torch.tensor(1.4, dtype=torch.float64)
    chi1 = torch.tensor(-0.25, dtype=torch.float64)
    chi2 = torch.tensor(0.6, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)

    mf, chif = zappa_remnant_properties(mass1, mass2, chi1, chi2, zero)
    mf_bbh_ratio, chif_bbh = bbh_remnant_uib2016(mass1, mass2, chi1, chi2)

    torch.testing.assert_close(mf, (mass1 + mass2) * mf_bbh_ratio, rtol=2e-14, atol=0.0)
    torch.testing.assert_close(chif, chif_bbh, rtol=2e-14, atol=0.0)
