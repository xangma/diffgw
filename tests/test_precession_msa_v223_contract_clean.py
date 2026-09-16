"""Small equation-level controls for the opt-in native MSA v223 branch."""
import math

import pytest
import torch

from torchwave.approximants.precession_msa_angles_clean import (
    _safe_cubic_roots,
    assemble_precession_msa_angles, assemble_precession_msa_angles_v223,
)
from torchwave.approximants.precession_msa_oscillatory_clean import (
    _branch_correction_v223,
    _P_v223,
    _partial_fractions,
    _partial_fractions_v223,
    branch_correction,
    oscillatory_correction,
    rational_coefficients,
    signed_nutation_rate,
)
from torchwave.approximants.precession_msa_pn_clean import (
    flux_coefficients,
    initial_reference_flux_coefficients,
    initial_reference_spin_couplings,
    psi2_compatibility,
    psi_coefficients,
)
from torchwave.approximants.precession_msa_reference_l_clean import (
    orbital_angular_momentum_nonconserved_3pn,
    orbital_angular_momentum_reference_3pn,
)
from torchwave.approximants.precession_msa_waveform_clean import (
    msa_final_spin,
)


MTSUN_SEC = 4.925490947641267e-6


def _anchor():
    return (
        torch.linspace(20.0, 200.0, 11, dtype=torch.float64),
        35.0,
        25.0,
        torch.tensor([0.2, 0.1, 0.2], dtype=torch.float64),
        torch.tensor([-0.1, 0.15, -0.1], dtype=torch.float64),
    )


def test_v223_reference_l_uses_a_distinct_259_over_75_coefficient():
    m1, m2, f, s1z, s2z = 35.0, 25.0, 20.0, 0.2, -0.1
    total = m1 + m2
    eta = m1 * m2 / total**2
    x = (math.pi * total * MTSUN_SEC * f) ** (2.0 / 3.0)
    spin_l = (m1 / total) ** 2 * s1z + (m2 / total) ** 2 * s2z
    opening = orbital_angular_momentum_nonconserved_3pn(
        m1, m2, f, s1z, s2z
    )
    reference = orbital_angular_momentum_reference_3pn(
        m1, m2, f, s1z, s2z
    )
    expected_delta = -(259.0 / 1800.0) * eta**2 * spin_l * x**2
    torch.testing.assert_close(
        reference - opening,
        torch.as_tensor(expected_delta, dtype=torch.float64),
        rtol=0.0,
        atol=2e-14,
    )


def test_v223_flux_uses_initial_reference_spin_contractions():
    eta = torch.tensor(35.0 * 25.0 / 60.0**2, dtype=torch.float64)
    q = torch.tensor(25.0 / 35.0, dtype=torch.float64)
    s1 = torch.tensor([0.2, 0.1, 0.2], dtype=torch.float64)
    s2 = torch.tensor([-0.1, 0.15, -0.1], dtype=torch.float64)
    n1, n2 = 35.0 / 60.0, 25.0 / 60.0
    S1, S2 = n1**2 * s1, n2**2 * s2
    couplings = initial_reference_spin_couplings(S1, S2)
    assert torch.allclose(couplings["s1l"], S1[2])
    assert torch.allclose(couplings["s2l"], S2[2])
    assert torch.allclose(couplings["s12"], torch.dot(S1, S2))
    expected = flux_coefficients(
        eta, q, S1[2], S2[2], torch.dot(S1, S1), torch.dot(S2, S2),
        torch.dot(S1, S2), S1[2] ** 2, S2[2] ** 2, S1[2] * S2[2],
    )
    actual = initial_reference_flux_coefficients(eta, q, S1, S2)
    for key in ("a0", "a2", "a3", "a4", "a5"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=1e-14)

    angles = assemble_precession_msa_angles(
        torch.tensor([20.0, 80.0], dtype=torch.float64), 35.0, 25.0,
        s1, s2, compatibility="v223",
    )
    assert angles["compatibility"] == "v223"
    for key in ("a0", "a2", "a3", "a4", "a5"):
        torch.testing.assert_close(
            angles["flux_coefficients"][key], actual[key], rtol=0.0, atol=2e-13
        )


def test_v223_output_selects_pn_beta_but_equation_output_stays_newtonian():
    frequency, _, _, s1, s2 = _anchor()
    native = assemble_precession_msa_angles(
        frequency, 35.0, 25.0, s1, s2, compatibility="v223",
    )
    equation = assemble_precession_msa_angles(
        frequency, 35.0, 25.0, s1, s2,
    )
    torch.testing.assert_close(
        native["cos_beta"], native["cos_beta_pn"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        equation["cos_beta"], equation["cos_beta_newtonian"], rtol=0.0, atol=0.0
    )
    assert torch.max(torch.abs(
        native["cos_beta"] - native["cos_beta_newtonian"]
    )) > 1.0e-8
    wrapped = assemble_precession_msa_angles_v223(
        frequency, 35.0, 25.0, s1, s2,
    )
    for key in ("alpha", "gamma", "cos_beta"):
        torch.testing.assert_close(native[key], wrapped[key], rtol=0.0, atol=0.0)


def test_v223_psi2_transformation_leaves_psi1_unchanged():
    values = [torch.tensor(v, dtype=torch.float64) for v in
              (0.24, 0.17, 0.71, 0.38, 0.12, 0.07, 0.04, 0.03, 2.3, -1.1)]
    independent = psi_coefficients(*values)
    compatible = psi_coefficients(*values, compatibility="v223")
    assert torch.allclose(compatible[0], independent[0], rtol=0.0, atol=0.0)
    expected = psi2_compatibility(independent[1], values[2], values[8], values[9])
    torch.testing.assert_close(compatible[1], expected, rtol=0.0, atol=1e-14)


def test_v223_oscillatory_signs_and_stable_branch_are_explicit():
    psi = torch.tensor(math.pi / 2.0, dtype=torch.float64, requires_grad=True)
    branch = branch_correction(torch.tensor(0.2, dtype=torch.float64), psi)
    assert torch.isfinite(branch)
    grad = torch.autograd.grad(branch, psi)[0]
    assert torch.isfinite(grad)

    # These denominator coefficients have poles lambda_1=.2 and lambda_2=-.5.
    args = dict(
        psi=torch.tensor(0.37, dtype=torch.float64),
        J=torch.tensor(0.83, dtype=torch.float64),
        psidot=torch.tensor(0.41, dtype=torch.float64),
        a=torch.tensor(1.0, dtype=torch.float64),
        c0=torch.tensor(0.1, dtype=torch.float64),
        c2=torch.tensor(0.2, dtype=torch.float64),
        c4=torch.tensor(0.3, dtype=torch.float64),
        d0=torch.tensor(1.0, dtype=torch.float64),
        d2=torch.tensor(0.3, dtype=torch.float64),
        d4=torch.tensor(-0.1, dtype=torch.float64),
        atheta=torch.tensor(0.7, dtype=torch.float64),
        btheta=torch.tensor(0.2, dtype=torch.float64),
    )
    equation = oscillatory_correction(**args)
    native = oscillatory_correction(**args, compatibility="v223")
    native_neg = oscillatory_correction(
        **{**args, "psidot": -args["psidot"]}, compatibility="v223"
    )
    assert torch.allclose(native[0], native_neg[0], rtol=0.0, atol=1e-14)
    assert torch.allclose(native[1], native_neg[1], rtol=0.0, atol=1e-14)
    assert all(torch.isfinite(value) for value in native)

    l1, l2, residue1, residue2, _ = _partial_fractions(
        args["c0"], args["c2"], args["c4"],
        args["d0"], args["d2"], args["d4"],
    )
    p1, p2 = _P_v223(l1, args["psi"]), _P_v223(l2, args["psi"])
    expected_phi = args["J"] / args["psidot"].abs() * (
        -residue1.abs() * p1 + residue2.abs() * p2
    )
    expected_zeta = args["J"] / args["psidot"].abs() * (
        -residue1.abs() * (args["atheta"] + args["btheta"] / l1) * p1
        + residue2.abs() * (args["atheta"] + args["btheta"] / l2) * p2
    )
    torch.testing.assert_close(native[0], expected_phi, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(native[1], expected_zeta, rtol=0.0, atol=1e-14)
    # The native zeta value excludes the standalone Q contribution retained
    # by the equation branch; this also guards against an accidental second J.
    assert not torch.allclose(native[1], equation[1], rtol=1e-8, atol=1e-10)
    assert torch.allclose(signed_nutation_rate(args["psidot"], v223=True),
                          -args["psidot"])


def test_v223_pole_labels_match_strict_algebraic_domain():
    """The continuation labels are the quadratic ``+``/``-`` roots."""
    values = (
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor(-0.1, dtype=torch.float64),
    )
    strict = _partial_fractions(*values)
    continued = _partial_fractions_v223(*values)
    for actual, expected in zip(continued[:4], strict[:4]):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_v223_negative_pole_factor_is_finite_and_keeps_second_sign():
    """Negative ``1-lambda`` continues through v223; equation stays strict."""
    args = dict(
        psi=torch.tensor(0.37, dtype=torch.float64),
        J=torch.tensor(0.83, dtype=torch.float64),
        psidot=torch.tensor(0.41, dtype=torch.float64),
        a=torch.tensor(1.0, dtype=torch.float64),
        c0=torch.tensor(0.1, dtype=torch.float64),
        c2=torch.tensor(0.2, dtype=torch.float64),
        c4=torch.tensor(0.3, dtype=torch.float64),
        d0=torch.tensor(1.0, dtype=torch.float64),
        d2=torch.tensor(-1.3, dtype=torch.float64),
        d4=torch.tensor(-0.1, dtype=torch.float64),
        atheta=torch.tensor(0.7, dtype=torch.float64),
        btheta=torch.tensor(0.2, dtype=torch.float64),
    )
    la, lb, ra, rb, _ = _partial_fractions_v223(
        args["c0"], args["c2"], args["c4"],
        args["d0"], args["d2"], args["d4"],
    )
    assert bool(la > 1.0)
    assert bool(lb < 0.0)
    assert bool(1.0 - la < 0.0)
    with pytest.raises(ValueError):
        oscillatory_correction(**args)

    native = oscillatory_correction(**args, compatibility="v223")
    assert all(torch.isfinite(value) for value in native)
    # Spell out the contract signs, including the signed second-pole
    # denominator factor, so an accidental positivity relabel is observable.
    ba = _branch_correction_v223(la, args["psi"])
    bb = _branch_correction_v223(lb, args["psi"])
    sign_a = torch.sign(la)
    sign_b = torch.sign(args["d0"] * (
        args["d0"] + args["d2"] + args["d4"]
    )) * torch.sign(lb)
    expected_phi = args["J"] / args["psidot"].abs() * (
        sign_a * ra.abs() * ba + sign_b * rb.abs() * bb
    )
    expected_zeta = args["J"] / args["psidot"].abs() * (
        sign_a * ra.abs() * (args["atheta"] + args["btheta"] / la) * ba
        + sign_b * rb.abs() * (args["atheta"] + args["btheta"] / lb) * bb
    )
    torch.testing.assert_close(native[0], expected_phi, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(native[1], expected_zeta, rtol=0.0, atol=1e-14)

    # At the exact n=0 endpoint the inactive first-pole primitive and its
    # geometric factor have an explicit finite zero convention.
    endpoint = {**args, "d2": torch.tensor(-0.5, dtype=torch.float64),
                "d4": torch.tensor(-0.5, dtype=torch.float64)}
    endpoint_out = oscillatory_correction(**endpoint, compatibility="v223")
    assert all(torch.isfinite(value) for value in endpoint_out)


def test_v223_finite_cubic_overshoot_uses_clamped_endpoint():
    """Finite acos overshoot is an endpoint, while undefined input is invalid."""
    roots, valid, degenerate = _safe_cubic_roots(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    # p=-1 and raw=3*sqrt(3)/2>1: acos is evaluated at its clamped endpoint.
    assert bool(valid)
    assert not bool(degenerate)
    assert torch.isfinite(roots).all()


def test_v223_degenerate_rows_keep_secular_angles_and_suppress_nutation():
    frequency, _, _, s1, s2 = _anchor()
    cases = (
        (35.0, 35.0, s1, s2),
        (35.0, 25.0, torch.tensor([0.0, 0.0, 0.2]),
         torch.tensor([0.0, 0.0, -0.1])),
        (35.0, 25.0, torch.zeros(3), torch.zeros(3)),
    )
    for m1, m2, spin1, spin2 in cases:
        out = assemble_precession_msa_angles(
            frequency, m1, m2, spin1, spin2, compatibility="v223",
        )
        for key in ("alpha", "gamma", "cos_beta", "psi", "psi_initial",
                    "secular_phi", "secular_zeta", "oscillatory_phi",
                    "oscillatory_zeta"):
            assert torch.isfinite(out[key]).all()
        # A repeated initial root removes the nutation phase and the
        # oscillatory correction.  The secular solution remains physical.
        assert bool(out["degenerate"])
        # Eq. (51) sets the additive integration constant to zero.  In the
        # anchored representation this means psi_initial=P(v_ref), rather
        # than a zero total phase at the reference frequency.
        total = torch.as_tensor(m1 + m2, dtype=torch.float64)
        q = torch.as_tensor(min(m1, m2) / max(m1, m2), dtype=torch.float64)
        delta = (1.0 - q) / (1.0 + q)
        g0 = 1.0 / out["flux_coefficients"]["a0"]
        v = (math.pi * total * MTSUN_SEC * frequency).pow(1.0 / 3.0)
        expected_phase = -0.75 * g0 * delta * (
            1.0 + out["psi1"] * v + out["psi2"] * v.square()
        ) / v.pow(3)
        torch.testing.assert_close(out["psi"], expected_phase, rtol=0.0, atol=3e-13)
        v_ref = (math.pi * total * MTSUN_SEC * 20.0).pow(1.0 / 3.0)
        expected_initial = -0.75 * g0 * delta * (
            1.0 + out["psi1"] * v_ref + out["psi2"] * v_ref.square()
        ) / v_ref.pow(3)
        torch.testing.assert_close(out["psi_initial"], expected_initial,
                                   rtol=0.0, atol=3e-13)
        assert torch.allclose(out["oscillatory_phi"], torch.zeros_like(out["oscillatory_phi"]))
        assert torch.allclose(out["oscillatory_zeta"], torch.zeros_like(out["oscillatory_zeta"]))
        assert torch.allclose(out["alpha"], out["secular_phi"], rtol=0.0, atol=2e-13)
        assert torch.allclose(out["gamma"], -out["secular_zeta"], rtol=0.0, atol=2e-13)
        assert torch.abs(out["secular_phi"][-1]) > 1.0e-8
        assert torch.abs(out["secular_zeta"][-1]) > 1.0e-8

    # Exact aligned and zero-spin rows use the audited limiting root triple;
    # the repeated-root threshold alone does not select this prescription.
    for m1, m2, spin1, spin2 in cases[1:]:
        out = assemble_precession_msa_angles(
            frequency, m1, m2, spin1, spin2, compatibility="v223",
        )
        total = torch.as_tensor(m1 + m2, dtype=torch.float64)
        n1, n2 = torch.as_tensor(m1, dtype=torch.float64) / total, torch.as_tensor(m2, dtype=torch.float64) / total
        ssum = n1.square() * spin1.to(dtype=torch.float64) + n2.square() * spin2.to(dtype=torch.float64)
        level = torch.sum(ssum.square())
        level = level.to(dtype=out["roots_reference"].dtype)
        expected = torch.stack((torch.zeros_like(level), level, level + 1.0e-9))
        torch.testing.assert_close(out["roots_reference"], expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(out["roots"], expected.expand_as(out["roots"]), rtol=0.0, atol=0.0)

    canonical = assemble_precession_msa_angles(
        frequency, 35.0, 25.0, s1, s2, compatibility="v223",
    )
    swapped = assemble_precession_msa_angles(
        frequency, 25.0, 35.0, s2, s1, compatibility="v223",
    )
    for key in ("alpha", "gamma", "cos_beta"):
        torch.testing.assert_close(canonical[key], swapped[key], rtol=2e-12, atol=2e-13)


def test_v223_current_gap_transition_uses_upper_root_and_keeps_series_phase():
    frequency = torch.arange(185.0, 197.0, dtype=torch.float64)
    m1, m2 = 35.0, 25.0
    spin1 = torch.tensor([1.0e-5, 0.0, 0.2], dtype=torch.float64)
    spin2 = torch.tensor([0.0, 1.0e-5, -0.1], dtype=torch.float64)
    out = assemble_precession_msa_angles(
        frequency, m1, m2, spin1, spin2, compatibility="v223",
    )

    # This exact fixture crosses the published current-frequency gap test at
    # 190 Hz.  The initial branch remains degenerate throughout.
    expected_degenerate = torch.tensor(
        [True, True, True, True, True, False, False, False, False, False,
         False, False], dtype=torch.bool
    )
    torch.testing.assert_close(out["frequency_degenerate"], expected_degenerate)
    assert bool(out["degenerate"])
    torch.testing.assert_close(
        out["spin_squared"][expected_degenerate],
        out["roots"][expected_degenerate, 2], rtol=0.0, atol=0.0,
    )

    # Current degeneracy suppresses only the oscillatory correction.  The
    # secular angles and the unanchored Eq. (51) series continue through the
    # transition, including the first ordinary-root frequency.
    total = torch.as_tensor(m1 + m2, dtype=torch.float64)
    q = torch.as_tensor(min(m1, m2) / max(m1, m2), dtype=torch.float64)
    delta = (1.0 - q) / (1.0 + q)
    g0 = 1.0 / out["flux_coefficients"]["a0"]
    v = (math.pi * total * MTSUN_SEC * frequency).pow(1.0 / 3.0)
    expected_phase = -0.75 * g0 * delta * (
        1.0 + out["psi1"] * v + out["psi2"] * v.square()
    ) / v.pow(3)
    torch.testing.assert_close(out["psi"], expected_phase, rtol=0.0, atol=3e-13)
    torch.testing.assert_close(
        out["oscillatory_phi"][expected_degenerate],
        torch.zeros_like(out["oscillatory_phi"][expected_degenerate]),
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        out["oscillatory_zeta"][expected_degenerate],
        torch.zeros_like(out["oscillatory_zeta"][expected_degenerate]),
        rtol=0.0, atol=0.0,
    )
    assert torch.isfinite(out["oscillatory_phi"]).all()
    assert torch.isfinite(out["oscillatory_zeta"]).all()
    # The first ordinary-root row has a negative pole factor, which is a
    # finite v223 continuation and must not be classified as rational invalid.
    assert not bool(out["rational_invalid_current"][5])
    torch.testing.assert_close(
        out["alpha"][expected_degenerate],
        out["secular_phi"][expected_degenerate], rtol=0.0, atol=2e-13,
    )
    torch.testing.assert_close(
        out["gamma"][expected_degenerate],
        -out["secular_zeta"][expected_degenerate], rtol=0.0, atol=2e-13,
    )


def test_v223_final_spin_uses_initial_total_mass_squared_lfin():
    sav2 = torch.tensor(0.13, dtype=torch.float64)
    mean1, mean2 = (torch.tensor(v, dtype=torch.float64) for v in (0.08, -0.02))
    initial1, initial2 = (torch.tensor(v, dtype=torch.float64) for v in (0.12, -0.04))
    aligned, mf = (torch.tensor(v, dtype=torch.float64) for v in (0.61, 0.93))
    lfin = aligned - initial1 - initial2
    expected_squared = sav2 + lfin.square() + 2.0 * lfin * (mean1 + mean2)
    actual = msa_final_spin(
        sav2, mean1, mean2, aligned, mf,
        normalization="total", average_is_squared=True,
        initial_s1_l=initial1, initial_s2_l=initial2,
        l_fin_normalization="initial",
    )
    torch.testing.assert_close(
        actual, torch.sqrt(expected_squared), rtol=0.0, atol=1e-14
    )
