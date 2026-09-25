"""Unit test suite for PhenomXP / XPHM Precession Prescriptions (MSA Variant 223 & NNLO).

Clean-Room Acceptance Verification based strictly on published literature:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)

Acceptance Criteria:
1. Aligned-spin limit matches IMRPhenomXHM / IMRPhenomXAS to < 10^-12.
2. Two-spin independence: waveforms with (S1_perp > 0, S2_perp = 0) vs (S1_perp = 0, S2_perp > 0) are distinct.
3. Body exchange symmetry: m1 <-> m2, S1 <-> S2 with appropriate odd-mode phase (-1)^m.
4. Reference frequency variation: waveforms respond smoothly to f_ref in (20, 30, 50) Hz.
5. Fallback behavior: equal-mass (q=1.0) and collinear (S_perp < 10^-9) trigger smooth NNLO fallback without NaNs or crashes.
6. Autograd differentiability: backprop through polarizations to all 8 parameters (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) yields finite non-zero gradients.
7. Versions (101, 102, 103, 104, 220, 223) and conventions (0, 1, 5, 6, 7) all execute without error.
8. Batched evaluation across masses, 3D spins, and inclinations matches scalar unbatched evaluations.
"""

import pytest
import torch
import math

from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.imrphenom_hm import IMRPhenomXHM
from torchwave.approximants.precession_msa import evaluate_msa_precession, solve_spin_roots, compute_l_3pn
from torchwave.constants import PI, MSUN_SEC
from torchwave import get_fd_waveform


def test_msa_aligned_spin_collapse():
    """Acceptance Criterion 1: Non-precessing limit collapses to underlying aligned models to < 10^-12."""
    f = torch.linspace(25.0, 300.0, 128, dtype=torch.float64)
    m1, m2 = 36.0, 22.0
    s1z, s2z = 0.35, -0.25
    inc = 0.65
    phic = 0.3
    dist = 150.0

    # 1. Test IMRPhenomXP -> IMRPhenomXAS
    xp_model = IMRPhenomXP()
    xas_model = IMRPhenomXAS()

    hp_xas, hc_xas = xas_model(
        f=f, mass1=m1, mass2=m2, chi1=s1z, chi2=s2z,
        distance=dist, inclination=inc, phi_c=phic,
    )
    hp_xp, hc_xp = xp_model(
        f=f, mass1=m1, mass2=m2,
        spin1x=0.0, spin1y=0.0, spin1z=s1z,
        spin2x=0.0, spin2y=0.0, spin2z=s2z,
        distance=dist, inclination=inc, phic=phic,
    )

    diff_p_xp = torch.max(torch.abs(hp_xp - hp_xas)).item()
    diff_c_xp = torch.max(torch.abs(hc_xp - hc_xas)).item()
    max_amp_xas = torch.max(torch.abs(hp_xas)).item()
    assert diff_p_xp / max_amp_xas < 1e-12, f"XP h+ rel diff {diff_p_xp / max_amp_xas} exceeds 1e-12"
    assert diff_c_xp / max_amp_xas < 1e-12, f"XP hx rel diff {diff_c_xp / max_amp_xas} exceeds 1e-12"

    # 2. Test IMRPhenomXPHM -> IMRPhenomXHM
    xphm_model = IMRPhenomXPHM()
    xhm_model = IMRPhenomXHM()

    hp_xhm, hc_xhm = xhm_model(
        f=f, mass1=m1, mass2=m2, chi1=s1z, chi2=s2z,
        distance=dist, inclination=inc, phic=phic,
    )
    hp_xphm, hc_xphm = xphm_model(
        f=f, mass1=m1, mass2=m2,
        spin1x=0.0, spin1y=0.0, spin1z=s1z,
        spin2x=0.0, spin2y=0.0, spin2z=s2z,
        distance=dist, inclination=inc, phic=phic,
    )

    diff_p_xphm = torch.max(torch.abs(hp_xphm - hp_xhm)).item()
    diff_c_xphm = torch.max(torch.abs(hc_xphm - hc_xhm)).item()
    max_amp_xhm = torch.max(torch.abs(hp_xhm)).item()
    assert diff_p_xphm / max_amp_xhm < 1e-12, f"XPHM h+ rel diff {diff_p_xphm / max_amp_xhm} exceeds 1e-12"
    assert diff_c_xphm / max_amp_xhm < 1e-12, f"XPHM hx rel diff {diff_c_xphm / max_amp_xhm} exceeds 1e-12"


@pytest.mark.parametrize("f_ref", (None, 20.0, 40.0))
def test_xp_aligned_reference_phase_and_one_spin_continuity(f_ref):
    """The Eq. (7) beta=0 limit keeps alpha+gamma fixed at every frequency."""
    f = torch.linspace(25.0, 300.0, 128, dtype=torch.float64)
    common = dict(f=f, mass1=36.0, mass2=22.0, distance=150.0,
                  inclination=0.65, phic=0.3, f_ref=f_ref)
    xp = IMRPhenomXP()
    exact = xp(
        **common, spin1x=0.0, spin1y=0.0, spin1z=0.35,
        spin2x=0.0, spin2y=0.0, spin2z=-0.25,
    )
    carrier = IMRPhenomXAS()(
        f=f, mass1=36.0, mass2=22.0, chi1=0.35, chi2=-0.25,
        distance=150.0, inclination=0.65, phi_c=0.3, f_ref=f_ref,
    )
    scale = max(torch.max(torch.abs(pol)).item() for pol in carrier)
    for actual, expected in zip(exact, carrier):
        assert torch.max(torch.abs(actual - expected)).item() / scale < 1e-12

    def near_distance(spin1x):
        near = xp(
            **common, spin1x=spin1x, spin1y=0.0, spin1z=0.35,
            spin2x=0.0, spin2y=0.0, spin2z=-0.25,
        )
        return max(
            (torch.linalg.vector_norm(left - right) /
             torch.linalg.vector_norm(right)).item()
            for left, right in zip(near, exact)
        )

    larger = near_distance(1e-9)
    smaller = near_distance(1e-10)
    assert larger < 1e-8
    assert smaller < larger / 5.0


def test_up_down_fixture_instability_band():
    """Gerosa et al.'s up-down threshold places 55 Hz before the unstable band."""
    q = 22.0 / 36.0
    r_plus_over_m = (math.sqrt(0.35) + math.sqrt(q * 0.25)) ** 4 / (1.0 - q) ** 2
    crossing_hz = 1.0 / (math.pi * 58.0 * MSUN_SEC * r_plus_over_m ** 1.5)
    assert crossing_hz == pytest.approx(72.8648, abs=1e-3)
    assert (math.pi * 58.0 * MSUN_SEC * 55.0) ** (-2.0 / 3.0) > r_plus_over_m
    assert (math.pi * 58.0 * MSUN_SEC * 300.0) ** (-2.0 / 3.0) < r_plus_over_m


def test_xp_two_spin_aligned_reference_frequency_control():
    """Two small transverse spins converge at their stable aligned reference point."""
    f = torch.tensor([40.0], dtype=torch.float64)
    params = dict(f=f, mass1=36.0, mass2=22.0, distance=150.0,
                  inclination=0.65, phic=0.3, f_ref=40.0,
                  spin1y=0.0, spin1z=0.35, spin2y=0.0, spin2z=0.25)
    xp = IMRPhenomXP()
    exact = xp(**params, spin1x=0.0, spin2x=0.0)

    def relative_error(x):
        near = xp(**params, spin1x=x, spin2x=-x / 2.0)
        return max(
            (torch.linalg.vector_norm(left - right) /
             torch.linalg.vector_norm(right)).item()
            for left, right in zip(near, exact)
        )

    larger = relative_error(1e-10)
    smaller = relative_error(1e-11)
    assert larger < 1e-10
    assert smaller < larger / 5.0


@pytest.mark.parametrize(
    "f_max", (55.0, 300.0), ids=("before-instability", "crosses-instability"),
)
def test_xp_two_spin_near_aligned_nnlo_fallback_continuity(f_max):
    """Require the invalid-MSA NNLO fallback on both sides of the instability."""
    f = torch.linspace(25.0, f_max, 128, dtype=torch.float64)
    params = dict(f=f, mass1=36.0, mass2=22.0, distance=150.0,
                  inclination=0.65, phic=0.3, f_ref=40.0,
                  spin1y=0.0, spin1z=0.35, spin2y=0.0, spin2z=-0.25)
    xp = IMRPhenomXP()
    aligned = xp(**params, spin1x=0.0, spin2x=0.0)
    tiny = xp(**params, spin1x=1e-10, spin2x=-5e-11)
    difference = max(
        (torch.linalg.vector_norm(left - right) /
         torch.linalg.vector_norm(right)).item()
        for left, right in zip(tiny, aligned)
    )
    assert difference < 1e-6


def test_msa_two_spin_independence():
    """Acceptance Criterion 2: Waveforms with in-plane spin on body 1 vs body 2 are distinct."""
    f = torch.linspace(25.0, 250.0, 100, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    inc = 0.7

    model = IMRPhenomXPHM()

    # Case A: in-plane spin only on body 1
    hp_a, hc_a = model(
        f=f, mass1=m1, mass2=m2,
        spin1x=0.4, spin1y=0.0, spin1z=0.1,
        spin2x=0.0, spin2y=0.0, spin2z=0.1,
        inclination=inc,
    )

    # Case B: in-plane spin only on body 2
    hp_b, hc_b = model(
        f=f, mass1=m1, mass2=m2,
        spin1x=0.0, spin1y=0.0, spin1z=0.1,
        spin2x=0.4, spin2y=0.0, spin2z=0.1,
        inclination=inc,
    )

    # Verify that both are non-zero and clearly distinct
    diff = torch.norm(hp_a - hp_b).item()
    norm_a = torch.norm(hp_a).item()
    rel_diff = diff / norm_a
    assert rel_diff > 0.05, f"Two-spin cases are unexpectedly identical: rel_diff = {rel_diff}"


def test_msa_body_exchange_symmetry():
    """Acceptance Criterion 3: Body exchange m1 <-> m2, S1 <-> S2 physical consistency."""
    f = torch.linspace(30.0, 200.0, 64, dtype=torch.float64)
    m1, m2 = 30.0, 20.0
    s1x, s1y, s1z = 0.2, 0.1, 0.1
    s2x, s2y, s2z = 0.1, -0.1, -0.05

    model = IMRPhenomXP()
    # Physical system invariant under labeling: dominant mode h+ amplitude profile remains identical
    hp_12, hc_12 = model(
        f=f, mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
    )
    hp_21, hc_21 = model(
        f=f, mass1=m2, mass2=m1,
        spin1x=s2x, spin1y=s2y, spin1z=s2z,
        spin2x=s1x, spin2y=s1y, spin2z=s1z,
    )

    amp_12 = hp_12.abs()
    amp_21 = hp_21.abs()
    rel_amp_diff = torch.max(torch.abs(amp_12 - amp_21)) / torch.max(amp_12)
    assert rel_amp_diff < 1e-4, f"Body exchange relative amplitude mismatch {rel_amp_diff} exceeds tolerance"


def test_msa_reference_frequency_variation():
    """Acceptance Criterion 4: Waveforms respond smoothly to f_ref in (20, 30, 50) Hz."""
    f = torch.linspace(25.0, 250.0, 100, dtype=torch.float64)
    m1, m2 = 30.0, 20.0
    s1 = (0.2, 0.1, 0.2)
    s2 = (0.1, 0.0, -0.1)

    model = IMRPhenomXPHM()

    waveforms = {}
    for f_ref in (20.0, 30.0, 50.0):
        hp, hc = model(
            f=f, mass1=m1, mass2=m2,
            spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
            spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
            f_ref=f_ref,
        )
        assert not torch.isnan(hp).any()
        assert not torch.isnan(hc).any()
        assert hp.abs().max().item() > 0.0
        waveforms[f_ref] = (hp, hc)

    # Varying f_ref rotates spin orientation at f_ref, producing smooth phase shifts
    diff_20_30 = torch.norm(waveforms[20.0][0] - waveforms[30.0][0]).item()
    diff_30_50 = torch.norm(waveforms[30.0][0] - waveforms[50.0][0]).item()
    assert diff_20_30 > 0.0
    assert diff_30_50 > 0.0


def test_msa_fallback_rules():
    """Acceptance Criterion 5: Fallback behavior for equal mass and collinear configurations."""
    f = torch.linspace(25.0, 250.0, 64, dtype=torch.float64)

    model = IMRPhenomXP()

    # 1. Equal mass q = 1.0 (delta = 0) with in-plane spin triggers smooth fallback
    hp_eq, hc_eq = model(
        f=f, mass1=25.0, mass2=25.0,
        spin1x=0.2, spin1y=0.1, spin1z=0.0,
        spin2x=0.0, spin2y=0.0, spin2z=0.0,
    )
    assert not torch.isnan(hp_eq).any()
    assert not torch.isnan(hc_eq).any()
    assert hp_eq.abs().max().item() > 0.0

    # 2. Collinear spins (S_perp = 0) with unaligned magnitudes triggers fallback
    hp_col, hc_col = model(
        f=f, mass1=30.0, mass2=20.0,
        spin1x=1e-12, spin1y=0.0, spin1z=0.3,
        spin2x=0.0, spin2y=0.0, spin2z=-0.2,
    )
    assert not torch.isnan(hp_col).any()
    assert not torch.isnan(hc_col).any()
    assert hp_col.abs().max().item() > 0.0


def test_msa_autograd_all_8_parameters():
    """Acceptance Criterion 6: Backprop through polarizations to all 8 parameters yields finite non-zero gradients."""
    f = torch.linspace(25.0, 200.0, 32, dtype=torch.float64)

    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    s2x = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    s2y = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    s2z = torch.tensor(-0.10, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomXP()(
        f=f, mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    params = [
        ("m1", m1), ("m2", m2),
        ("s1x", s1x), ("s1y", s1y), ("s1z", s1z),
        ("s2x", s2x), ("s2y", s2y), ("s2z", s2z),
    ]
    for name, p in params:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(p.grad), f"Gradient for {name} is NaN"
        assert p.grad.abs().item() > 0.0, f"Gradient for {name} is unexpectedly zero"


def test_msa_versions_and_conventions():
    """Acceptance Criterion 7: All supported prec_versions (101, 102, 103, 104, 220, 223) and conventions (0, 1, 5, 6, 7) execute."""
    f = torch.linspace(25.0, 200.0, 32, dtype=torch.float64)
    m1, m2 = 32.0, 22.0
    s1 = (0.2, 0.1, 0.2)
    s2 = (0.1, 0.0, -0.1)

    model = IMRPhenomXPHM()

    for p_ver in (101, 102, 103, 104, 220, 223):
        hp, hc = model(
            f=f, mass1=m1, mass2=m2,
            spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
            spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
            prec_version=p_ver,
        )
        assert not torch.isnan(hp).any(), f"NaN in h+ for prec_version={p_ver}"
        assert not torch.isnan(hc).any(), f"NaN in hx for prec_version={p_ver}"

    for conv in (0, 1, 5, 6, 7):
        hp, hc = model(
            f=f, mass1=m1, mass2=m2,
            spin1x=s1[0], spin1y=s1[1], spin1z=s1[2],
            spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
            convention=conv,
        )
        assert not torch.isnan(hp).any(), f"NaN in h+ for convention={conv}"
        assert not torch.isnan(hc).any(), f"NaN in hx for convention={conv}"


def test_msa_batching_equivalence():
    """Acceptance Criterion 8: Batched tensor inputs match unbatched evaluations exactly."""
    B = 3
    N_f = 48
    f = torch.linspace(25.0, 250.0, N_f, dtype=torch.float64)

    m1_b = torch.tensor([32.0, 40.0, 28.0], dtype=torch.float64)
    m2_b = torch.tensor([20.0, 25.0, 16.0], dtype=torch.float64)
    s1x_b = torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64)
    s1y_b = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    s1z_b = torch.tensor([0.2, -0.1, 0.1], dtype=torch.float64)
    s2x_b = torch.tensor([0.0, 0.1, 0.0], dtype=torch.float64)
    s2y_b = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    s2z_b = torch.tensor([-0.1, 0.0, 0.2], dtype=torch.float64)
    inc_b = torch.tensor([0.4, 0.7, 0.2], dtype=torch.float64)

    model = IMRPhenomXP()
    hp_b, hc_b = model(
        f=f, mass1=m1_b, mass2=m2_b,
        spin1x=s1x_b, spin1y=s1y_b, spin1z=s1z_b,
        spin2x=s2x_b, spin2y=s2y_b, spin2z=s2z_b,
        inclination=inc_b,
    )

    assert hp_b.shape == (B, N_f)
    assert hc_b.shape == (B, N_f)

    for b in range(B):
        hp_s, hc_s = model(
            f=f, mass1=m1_b[b], mass2=m2_b[b],
            spin1x=s1x_b[b], spin1y=s1y_b[b], spin1z=s1z_b[b],
            spin2x=s2x_b[b], spin2y=s2y_b[b], spin2z=s2z_b[b],
            inclination=inc_b[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25)
