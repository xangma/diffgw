"""Separate the v223 oracle chart from the aligned fixed-basis limit.

Pratten et al., arXiv:2004.06503v2, Eqs. (8)--(13) and Appendix C
Eqs. (67)--(81), specify the rotation and observer frame. The finite-beta
floor and exact-zero chart are explicitly v223 compatibility behavior.
"""

import math

import pytest
import torch

from torchwave.approximants.precession_msa_angles_clean import assemble_precession_msa_angles
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry


FREQUENCIES = torch.tensor([25.0, 40.0, 80.0], dtype=torch.float64)


def _relative_l2(left, right):
    left = torch.stack(left)
    right = torch.stack(right)
    return torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right)


def _clean(spin1x, spin1y):
    return waveform_v223(
        FREQUENCIES, 40.0, 20.0, spin1x, spin1y, 0.4, 0.0, 0.0, -0.2,
        distance=100.0, inclination=0.7, phic=0.3, f_ref=40.0,
        convention=1, final_spin_mod=3,
    )


def _geometry(spin1x, spin1y):
    return source_frame_geometry(
        40.0, 20.0, spin1x, spin1y, 0.4, 0.0, 0.0, -0.2,
        0.7, 0.3, 40.0, compatibility="v223",
    )


@pytest.mark.parametrize("azimuth", (-2.0, 0.4, 2.1))
def test_v223_appendix_c_combined_chart_has_a_continuous_aligned_limit(azimuth):
    # phi_J itself is undefined at J_perp=0; its limiting value follows the
    # transverse spin azimuth. The exact v223 endpoint chart adds pi to
    # alpha0, after which the effective alpha0-phi_J has a fixed limit.
    exact_alpha, _, _, exact_phi_j = _geometry(0.0, 0.0)
    near_alpha, _, _, near_phi_j = _geometry(
        1.0e-9 * math.cos(azimuth), 1.0e-9 * math.sin(azimuth),
    )
    torch.testing.assert_close(
        torch.exp(1j * (near_alpha - near_phi_j)),
        torch.exp(1j * (exact_alpha + math.pi - exact_phi_j)),
        rtol=2e-9, atol=2e-9,
    )



def test_v223_split_root_has_nonzero_aligned_opening():
    # A nonzero beta at exact spin alignment lets the uncorrected v223 Euler
    # chart leave a finite, azimuth-dependent contribution after projection.
    angles = assemble_precession_msa_angles(
        FREQUENCIES, 40.0, 20.0,
        torch.tensor([0.0, 0.0, 0.4], dtype=torch.float64),
        torch.tensor([0.0, 0.0, -0.2], dtype=torch.float64),
        f_ref=40.0, compatibility="v223",
    )
    assert torch.max(1.0 - angles["cos_beta"]) > 1.0e-10


def test_lal_v223_near_aligned_chart_matches_clean_but_not_fixed_basis_limit():
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 223)
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(options, 5)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(options, 1)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 3)

    def oracle(spin1x, spin1y):
        hp, hc = ls.SimInspiralChooseFDWaveform(
            40.0 * lal.MSUN_SI, 20.0 * lal.MSUN_SI,
            spin1x, spin1y, 0.4, 0.0, 0.0, -0.2,
            100.0e6 * lal.PC_SI, 0.7, 0.3, 0.0, 0.0, 0.0,
            0.25, 20.0, 81.0, 40.0, options, ls.IMRPhenomXP,
        )
        return tuple(
            torch.as_tensor(series.data.data[[100, 160, 320]].copy(), dtype=torch.complex128)
            for series in (hp, hc)
        )

    exact_clean = _clean(0.0, 0.0)
    exact_oracle = oracle(0.0, 0.0)
    assert _relative_l2(exact_clean, exact_oracle) < 1.0e-8
    for azimuth in (-2.0, 0.4, 2.1):
        spin1x = 1.0e-9 * math.cos(azimuth)
        spin1y = 1.0e-9 * math.sin(azimuth)
        near_clean = _clean(spin1x, spin1y)
        near_oracle = oracle(spin1x, spin1y)
        assert _relative_l2(near_clean, near_oracle) < 1.0e-8
        # The oracle itself violates the 2e-8 fixed-basis continuity gate.
        # This is a diagnostic of incompatible contracts, not an acceptance
        # tolerance for the physical near-aligned limit.
        assert _relative_l2(near_oracle, exact_oracle) > 2.0e-8
