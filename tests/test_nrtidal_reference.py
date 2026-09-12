"""Independent acceptance tests for reference-normalized self-spin phase.

Derived strictly from published literature (Dietrich et al. 2019, arXiv:1905.06011v2 Eqs. 26-29)
and the compatibility specification in SPECIFICATION.md.
No oracle sources, no external ports, no LAL/PyCBC imports.
"""

import math
import torch

from torchwave.constants import PI, MSUN_SEC
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
    _nrtidal_self_spin_components,
    _nrtidal_self_spin_correction,
    _bns_merger_frequency,
)


def _independent_paper_self_spin_components(x, m1, m2, s1, s2, l1, l2):
    """Independent evaluation of Dietrich et al. (2019) Eqs. 26-29 for S23 and S35."""
    m_tot = m1 + m2
    eta = (m1 * m2) / (m_tot * m_tot)
    xa = m1 / m_tot
    xb = m2 / m_tot
    def i_love_q(lam_val):
        lam_s = max(float(lam_val), 1.0)
        ln_l = math.log(lam_s)
        # Yagi & Yunes (2013) Science 341, 365 Eq. (2) / Table S1:
        ln_cq = 0.1940 + 0.0936 * ln_l + 0.0474 * (ln_l**2) - 0.00421 * (ln_l**3) + 0.000123 * (ln_l**4)
        cq = math.exp(ln_cq) if float(lam_val) > 1e-4 else 1.0
        ln_cq_s = max(ln_cq, 1e-8)
        ln_coc = 0.003131 + 2.071 * ln_cq_s - 0.7152 * (ln_cq_s**2) + 0.2458 * (ln_cq_s**3) - 0.03309 * (ln_cq_s**4)
        coc = math.exp(ln_coc) if float(lam_val) > 1e-4 else 1.0
        return cq, coc

    cq1, coc1 = i_love_q(l1)
    cq2, coc2 = i_love_q(l2)
    ch_q1, ch_oc1 = cq1 - 1.0, coc1 - 1.0
    ch_q2, ch_oc2 = cq2 - 1.0, coc2 - 1.0

    def star(x_a, x_b, ch_a, ch_b, cq_hat, coc_hat):
        p2 = -50.0 * cq_hat * (x_a**2) * (ch_a**2)
        p3 = (5.0 / 84.0) * (9407.0 + 8218.0 * x_a - 2016.0 * (x_a**2)) * cq_hat * (x_a**2) * (ch_a**2)
        bracket = (x_a**2 + (308.0 / 3.0) * x_a) * ch_a + (x_b**2 - (89.0 / 3.0) * x_b) * ch_b - 40.0 * math.pi
        p35 = 10.0 * bracket * cq_hat * (x_a**2) * (ch_a**2) - 440.0 * coc_hat * (x_a**3) * (ch_a**3)
        return p2, p3, p35

    p2_1, p3_1, p35_1 = star(xa, xb, s1, s2, ch_q1, ch_oc1)
    p2_2, p3_2, p35_2 = star(xb, xa, s2, s1, ch_q2, ch_oc2)
    p2 = p2_1 + p2_2
    p3 = p3_1 + p3_2
    p35 = p35_1 + p35_2

    # Vectorized across x
    x_t = torch.as_tensor(x, dtype=torch.float64)
    pref = 3.0 / (128.0 * eta * (x_t ** 2.5))
    s23 = pref * (p2 * (x_t ** 2.0) + p3 * (x_t ** 3.0))
    s35 = pref * (p35 * (x_t ** 3.5))
    return s23, s35


# 3 Physical parameter systems with different masses, spin signs, and tidal deformabilities
SYSTEMS = [
    # System 1: Standard BNS with unequal masses and opposite spins
    dict(m1=1.50, m2=1.25, s1=0.04, s2=-0.02, l1=320.0, l2=480.0),
    # System 2: Higher mass ratio, swapped body spins
    dict(m1=1.65, m2=1.15, s1=-0.03, s2=0.05, l1=180.0, l2=750.0),
    # System 3: Nearly equal mass, both positive spins
    dict(m1=1.38, m2=1.32, s1=0.02, s2=0.03, l1=420.0, l2=450.0),
]


def test_independent_self_spin_separation_matches_paper():
    """Verify that independent paper formula matches _nrtidal_self_spin_components."""
    f = torch.tensor([30.0, 100.0, 500.0, 1000.0], dtype=torch.float64)
    for p in SYSTEMS:
        m_sec = (p["m1"] + p["m2"]) * MSUN_SEC
        x = (PI * m_sec * f) ** (2.0 / 3.0)
        s23_paper, s35_paper = _independent_paper_self_spin_components(
            x, p["m1"], p["m2"], p["s1"], p["s2"], p["l1"], p["l2"]
        )
        s23_prod, s35_prod = _nrtidal_self_spin_components(
            x=x,
            mass1=torch.tensor(p["m1"], dtype=torch.float64),
            mass2=torch.tensor(p["m2"], dtype=torch.float64),
            chi1=torch.tensor(p["s1"], dtype=torch.float64),
            chi2=torch.tensor(p["s2"], dtype=torch.float64),
            lambda1=torch.tensor(p["l1"], dtype=torch.float64),
            lambda2=torch.tensor(p["l2"], dtype=torch.float64),
        )
        torch.testing.assert_close(s23_prod, s23_paper, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(s35_prod, s35_paper, rtol=1e-12, atol=1e-15)
        # Total helper sum
        total_prod = _nrtidal_self_spin_correction(
            x=x,
            mass1=torch.tensor(p["m1"], dtype=torch.float64),
            mass2=torch.tensor(p["m2"], dtype=torch.float64),
            chi1=torch.tensor(p["s1"], dtype=torch.float64),
            chi2=torch.tensor(p["s2"], dtype=torch.float64),
            lambda1=torch.tensor(p["l1"], dtype=torch.float64),
            lambda2=torch.tensor(p["l2"], dtype=torch.float64),
        )
        torch.testing.assert_close(total_prod, s23_paper + s35_paper, rtol=1e-12, atol=1e-15)


def test_reference_normalization_complex_ratio():
    """Verify the exact complex phase factor relating coalescence and reference waveforms.

    From the specification:
    deltaPsi_ref(f) = T(f) + S23(f) - S23(f_ref) + S35(f)
    deltaPsi_coal(f) = T(f) + S23(f) + S35(f)
    Difference: deltaPsi_coal(f) - deltaPsi_ref(f) = S23(f_ref)
    Waveform ratio: (h_coal / h_ref) = exp(-i * S23(f_ref)) * (h_bbh_coal / h_bbh_ref)
    Or equivalently on the tidal ratio (h / h_bbh):
    R_ref(f) / R_coal(f) = exp(i * S23(f_ref))
    """
    f = torch.linspace(40.0, 1000.0, 100, dtype=torch.float64)
    model_bbh = IMRPhenomD()
    model_bns = IMRPhenomD_NRTidalv2()

    f_refs = [20.0, 50.0, 100.0]

    for p in SYSTEMS:
        for f_ref in f_refs:
            kw = dict(
                mass1=p["m1"],
                mass2=p["m2"],
                chi1=p["s1"],
                chi2=p["s2"],
                distance=100.0,
                inclination=0.4,
                apply_merger_taper=False,
            )

            # 1. Coalescence mode (explicit or f_ref=None)
            hp_bbh_c, hc_bbh_c = model_bbh(f=f, phasing_convention="coalescence", **{k: kw[k] for k in kw if k not in ("apply_merger_taper", "lambda1", "lambda2")})
            hp_bns_c, hc_bns_c = model_bns(f=f, phasing_convention="coalescence", lambda1=p["l1"], lambda2=p["l2"], **kw)

            # 2. Reference mode at f_ref
            hp_bbh_r, hc_bbh_r = model_bbh(f=f, f_ref=f_ref, phasing_convention="reference_frequency", **{k: kw[k] for k in kw if k not in ("apply_merger_taper", "lambda1", "lambda2")})
            hp_bns_r, hc_bns_r = model_bns(f=f, f_ref=f_ref, phasing_convention="reference_frequency", lambda1=p["l1"], lambda2=p["l2"], **kw)

            # Tidal-to-BBH ratio for coalescence and reference
            ratio_c = hp_bns_c / hp_bbh_c
            ratio_r = hp_bns_r / hp_bbh_r

            # Independent paper S23 at f_ref
            m_sec = (p["m1"] + p["m2"]) * MSUN_SEC
            x_ref = (PI * m_sec * f_ref) ** (2.0 / 3.0)
            s23_ref, _ = _independent_paper_self_spin_components(
                torch.tensor([x_ref], dtype=torch.float64),
                p["m1"], p["m2"], p["s1"], p["s2"], p["l1"], p["l2"]
            )
            s23_ref_val = s23_ref.item()

            # Expected ratio: ratio_ref / ratio_coal = exp(i * S23(f_ref))
            expected_factor = torch.complex(
                torch.tensor(math.cos(s23_ref_val), dtype=torch.float64),
                torch.tensor(math.sin(s23_ref_val), dtype=torch.float64),
            )

            actual_factor = ratio_r / ratio_c
            torch.testing.assert_close(
                actual_factor,
                expected_factor.expand_as(actual_factor),
                rtol=1e-11,
                atol=1e-13,
            )


def test_body_swap_symmetry():
    """Verify exact body-swap symmetry for swapped masses, spins, and tides."""
    f = torch.linspace(40.0, 800.0, 50, dtype=torch.float64)
    model = IMRPhenomD_NRTidalv2()
    f_ref = 35.0

    hp1, hc1 = model(
        f=f,
        mass1=1.6,
        mass2=1.2,
        chi1=0.03,
        chi2=-0.02,
        lambda1=250.0,
        lambda2=600.0,
        f_ref=f_ref,
        phasing_convention="reference_frequency",
        apply_merger_taper=False,
    )
    hp2, hc2 = model(
        f=f,
        mass1=1.2,
        mass2=1.6,
        chi1=-0.02,
        chi2=0.03,
        lambda1=600.0,
        lambda2=250.0,
        f_ref=f_ref,
        phasing_convention="reference_frequency",
        apply_merger_taper=False,
    )

    scale = 1e-23
    torch.testing.assert_close(hp1 / scale, hp2 / scale, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(hc1 / scale, hc2 / scale, rtol=1e-10, atol=1e-12)


def test_tidal_and_35pn_not_reference_subtracted():
    """Verify that dynamical tidal T(f) and S35(f) are NOT subtracted at f_ref.

    The exact S23(f_ref) phase shift detects any unwanted nonzero tidal or S35
    reference subtraction: if T or S35 were subtracted, the phase shift between
    coalescence and reference would deviate from S23(f_ref). We independently
    evaluate S23(f_ref) and S35(f_ref) from published equations and assert that
    the phase difference equals S23(f_ref) and not S23(f_ref) + S35(f_ref).
    """
    f = torch.tensor([150.0], dtype=torch.float64)
    f_ref = 40.0
    p = SYSTEMS[0]
    model_bbh = IMRPhenomD()
    model_bns = IMRPhenomD_NRTidalv2()

    kw_bbh = dict(
        mass1=p["m1"], mass2=p["m2"], chi1=p["s1"], chi2=p["s2"],
        distance=100.0, inclination=0.0,
    )
    kw_bns = dict(
        mass1=p["m1"], mass2=p["m2"], chi1=p["s1"], chi2=p["s2"],
        lambda1=p["l1"], lambda2=p["l2"],
        distance=100.0, inclination=0.0, apply_merger_taper=False,
    )

    hp_bbh_c, _ = model_bbh(f=f, phasing_convention="coalescence", **kw_bbh)
    hp_bns_c, _ = model_bns(f=f, phasing_convention="coalescence", **kw_bns)

    hp_bbh_r, _ = model_bbh(f=f, f_ref=f_ref, phasing_convention="reference_frequency", **kw_bbh)
    hp_bns_r, _ = model_bns(f=f, f_ref=f_ref, phasing_convention="reference_frequency", **kw_bns)

    phase_c = torch.angle(hp_bns_c / hp_bbh_c).item()
    phase_r = torch.angle(hp_bns_r / hp_bbh_r).item()
    delta_phi = (phase_r - phase_c) % (2.0 * math.pi)
    if delta_phi > math.pi:
        delta_phi -= 2.0 * math.pi

    m_sec = (p["m1"] + p["m2"]) * MSUN_SEC
    x_ref = (PI * m_sec * f_ref) ** (2.0 / 3.0)
    s23_ref, s35_ref = _independent_paper_self_spin_components(
        torch.tensor([x_ref], dtype=torch.float64),
        p["m1"], p["m2"], p["s1"], p["s2"], p["l1"], p["l2"]
    )

    # The phase shift MUST equal S23(f_ref) within tolerance
    assert math.isclose(delta_phi, s23_ref.item(), rel_tol=1e-8, abs_tol=1e-10)

    # And MUST NOT equal S23 + S35 or 0
    assert not math.isclose(delta_phi, (s23_ref + s35_ref).item(), rel_tol=1e-3, abs_tol=1e-5)
    assert abs(s35_ref.item()) > 1e-6, "S35 must be non-trivial to verify non-subtraction"


def test_convention_aliases_and_override_hierarchy():
    """Verify priority: reference_frame overrides phasing_convention;
    f_ref present selects reference_frequency; absent selects coalescence.
    """
    f = torch.linspace(40.0, 500.0, 30, dtype=torch.float64)
    model = IMRPhenomD_NRTidalv2()
    kw = dict(mass1=1.45, mass2=1.35, chi1=0.03, chi2=-0.02, lambda1=350.0, lambda2=420.0, apply_merger_taper=False)
    scale = 1e-23

    # Baseline coalescence: neither f_ref nor convention given
    hp_coal, hc_coal = model(f=f, **kw)

    # Explicit coalescence with f_ref given but overridden by convention
    hp_c_expl, hc_c_expl = model(f=f, f_ref=50.0, phasing_convention="coalescence", **kw)
    torch.testing.assert_close(hp_c_expl / scale, hp_coal / scale, rtol=1e-12, atol=1e-15)

    # reference_frame overrides phasing_convention
    hp_override, hc_override = model(
        f=f,
        f_ref=50.0,
        phasing_convention="reference_frequency",
        reference_frame="coalescence",
        **kw,
    )
    torch.testing.assert_close(hp_override / scale, hp_coal / scale, rtol=1e-12, atol=1e-15)

    # Implicit reference frequency when f_ref is provided without convention
    hp_ref_implicit, hc_ref_implicit = model(f=f, f_ref=50.0, **kw)
    hp_ref_explicit, hc_ref_explicit = model(f=f, f_ref=50.0, phasing_convention="reference_frequency", **kw)
    torch.testing.assert_close(hp_ref_implicit / scale, hp_ref_explicit / scale, rtol=1e-12, atol=1e-15)

    # Aliases 'reffreq' and 'ref' match 'reference_frequency'
    for alias in ("reffreq", "ref", "REFERENCE_FREQUENCY", "RefFreq"):
        hp_alias, _ = model(f=f, f_ref=50.0, phasing_convention=alias, **kw)
        torch.testing.assert_close(hp_alias / scale, hp_ref_explicit / scale, rtol=1e-12, atol=1e-15)

    # Explicit reference without f_ref defaults to f[0] matching base model
    hp_default_ref, _ = model(f=f, phasing_convention="reference_frequency", **kw)
    hp_f0_ref, _ = model(f=f, f_ref=f[0].item(), phasing_convention="reference_frequency", **kw)
    torch.testing.assert_close(hp_default_ref / scale, hp_f0_ref / scale, rtol=1e-12, atol=1e-15)


def test_v1_invariance_under_reference_settings():
    """Verify v1 waveforms are invariant under reference settings (no self-spin terms in v1)."""
    f = torch.linspace(40.0, 500.0, 30, dtype=torch.float64)
    model = IMRPhenomD_NRTidal(version="v1")
    model_bbh = IMRPhenomD()
    kw_bbh = dict(mass1=1.4, mass2=1.3, chi1=0.03, chi2=-0.02)
    kw_v1 = dict(mass1=1.4, mass2=1.3, chi1=0.03, chi2=-0.02, lambda1=300.0, lambda2=400.0, apply_merger_taper=False)

    # In v1, tidal-to-BBH ratio must be identical between coalescence and reference
    hp_bbh_c, _ = model_bbh(f=f, phasing_convention="coalescence", **kw_bbh)
    hp_v1_c, _ = model(f=f, phasing_convention="coalescence", **kw_v1)

    hp_bbh_r, _ = model_bbh(f=f, f_ref=50.0, phasing_convention="reference_frequency", **kw_bbh)
    hp_v1_r, _ = model(f=f, f_ref=50.0, phasing_convention="reference_frequency", **kw_v1)

    ratio_c = hp_v1_c / hp_bbh_c
    ratio_r = hp_v1_r / hp_bbh_r
    torch.testing.assert_close(ratio_c, ratio_r, rtol=1e-12, atol=1e-15)


def test_zero_spin_and_zero_tide_recovery():
    """Verify zero-spin and zero-tide recovery under reference normalization."""
    f = torch.linspace(40.0, 500.0, 30, dtype=torch.float64)
    model_bns = IMRPhenomD_NRTidalv2()
    model_bbh = IMRPhenomD()
    scale = 1e-23

    # Zero-tide with spins: recovers BBH exactly
    kw_bbh = dict(mass1=1.4, mass2=1.3, chi1=0.04, chi2=-0.03, distance=100.0, f_ref=45.0)
    hp_bbh, hc_bbh = model_bbh(f=f, **kw_bbh)
    hp_bns_0tide, hc_bns_0tide = model_bns(f=f, lambda1=0.0, lambda2=0.0, apply_merger_taper=False, **kw_bbh)
    torch.testing.assert_close(hp_bns_0tide / scale, hp_bbh / scale, rtol=1e-11, atol=1e-14)
    torch.testing.assert_close(hc_bns_0tide / scale, hc_bbh / scale, rtol=1e-11, atol=1e-14)

    # Zero-spin with tides: S23 = 0, so reference-normalization has no self-spin subtraction
    kw_0s_bbh = dict(mass1=1.4, mass2=1.3, chi1=0.0, chi2=0.0, distance=100.0)
    kw_0s_bns = dict(mass1=1.4, mass2=1.3, chi1=0.0, chi2=0.0, lambda1=350.0, lambda2=420.0, distance=100.0, apply_merger_taper=False)
    hp_0s_bbh_c, _ = model_bbh(f=f, phasing_convention="coalescence", **kw_0s_bbh)
    hp_0s_bns_c, _ = model_bns(f=f, phasing_convention="coalescence", **kw_0s_bns)

    hp_0s_bbh_r, _ = model_bbh(f=f, f_ref=45.0, phasing_convention="reference_frequency", **kw_0s_bbh)
    hp_0s_bns_r, _ = model_bns(f=f, f_ref=45.0, phasing_convention="reference_frequency", **kw_0s_bns)

    ratio_0s_c = hp_0s_bns_c / hp_0s_bbh_c
    ratio_0s_r = hp_0s_bns_r / hp_0s_bbh_r
    torch.testing.assert_close(ratio_0s_c, ratio_0s_r, rtol=1e-12, atol=1e-15)


def test_vector_f_ref_and_batched_parity():
    """Verify scalar f_ref matches vector f_ref across batches."""
    f = torch.linspace(30.0, 1000.0, 50, dtype=torch.float64)
    model = IMRPhenomD_NRTidalv2()

    m1 = torch.tensor([1.50, 1.65, 1.38], dtype=torch.float64)
    m2 = torch.tensor([1.25, 1.15, 1.32], dtype=torch.float64)
    s1 = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float64)
    s2 = torch.tensor([-0.02, 0.05, 0.03], dtype=torch.float64)
    l1 = torch.tensor([320.0, 180.0, 420.0], dtype=torch.float64)
    l2 = torch.tensor([480.0, 750.0, 450.0], dtype=torch.float64)
    f_refs = torch.tensor([25.0, 50.0, 75.0], dtype=torch.float64)

    hp_batched, hc_batched = model(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        lambda1=l1,
        lambda2=l2,
        f_ref=f_refs,
        phasing_convention="reference_frequency",
        apply_merger_taper=False,
    )

    scale = 1e-23
    for i in range(3):
        hp_scalar, hc_scalar = model(
            f=f,
            mass1=m1[i],
            mass2=m2[i],
            chi1=s1[i],
            chi2=s2[i],
            lambda1=l1[i],
            lambda2=l2[i],
            f_ref=f_refs[i],
            phasing_convention="reference_frequency",
            apply_merger_taper=False,
        )
        torch.testing.assert_close(hp_batched[i] / scale, hp_scalar / scale, rtol=1e-11, atol=1e-14)
        torch.testing.assert_close(hc_batched[i] / scale, hc_scalar / scale, rtol=1e-11, atol=1e-14)


def test_autograd_full_complex_gradcheck_including_f_ref():
    """Verify autograd gradcheck with f_ref, masses, spins, tides, distance, inclination, and DC/cutoff bins."""
    m1_val, m2_val = 1.45, 1.35
    l1_val, l2_val = 350.0, 420.0
    fm = _bns_merger_frequency(
        torch.tensor(m1_val, dtype=torch.float64),
        torch.tensor(m2_val, dtype=torch.float64),
        torch.tensor(l1_val, dtype=torch.float64),
        torch.tensor(l2_val, dtype=torch.float64),
    ).item()

    # Include DC bin (0.0), negative frequency (-1.0), in-band frequencies, taper window, and beyond cutoff
    f = torch.tensor([-1.0, 0.0, 30.0, 150.0, 0.95 * fm, 1.05 * fm, 1.3 * fm, 1e5], dtype=torch.float64)
    model = IMRPhenomD_NRTidalv2()

    # Parameters: [m1, m2, s1, s2, l1_norm, l2_norm, dist_norm, inc, f_ref]
    p = torch.tensor([1.45, 1.35, 0.02, -0.015, 0.875, 1.05, 1.0, 0.45, 35.0], dtype=torch.float64, requires_grad=True)

    def fn(params):
        hp, hc = model(
            f=f,
            mass1=params[0],
            mass2=params[1],
            chi1=params[2],
            chi2=params[3],
            lambda1=400.0 * params[4],
            lambda2=400.0 * params[5],
            distance=100.0 * params[6],
            inclination=params[7],
            f_ref=params[8],
            phasing_convention="reference_frequency",
        )
        return torch.view_as_real(torch.stack((hp, hc)) / 1e-23)

    out = fn(p)
    assert torch.isfinite(out).all()
    # DC and non-positive frequencies must be exactly zero
    assert (out[:, [0, 1]] == 0.0).all()

    # Run gradcheck on active frequencies
    assert torch.autograd.gradcheck(fn, (p,), eps=1e-7, atol=5e-5, rtol=5e-4)
