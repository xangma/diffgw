# SPDX-License-Identifier: MIT
"""Equation-level tests for the clean XHM inspiral amplitude assembly."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.diagnose_xhm_inspiral_assembly_clean import (
    final_spin_and_f_isco,
    frequencies_below,
    geometric_amplitudes_from_common_h,
    provider_collocations,
    chi_pn,
    validate_frequency_support,
    validate_veto_support,
)
from torchwave.approximants.xhm_fits import XHMParameterFits

from torchwave.approximants.xhm_amplitude_clean import (
    normalized_pn_amplitude,
    normalized_pn_amplitude_compatibility,
)
from torchwave.approximants.xhm_inspiral_assembly_clean import (
    COLLOCATION_FRACTIONS,
    COMPATIBILITY_CUT_VERSION,
    PRINTED_CUT_VERSION,
    compatibility_inspiral_cut_frequency,
    inspiral_cut_frequency,
    normalized_inspiral_amplitude,
    physical_inspiral_amplitude,
    printed_inspiral_cut_frequency,
    printed_veto_nodes,
)
from torchwave.approximants.xhm_spa_clean import (
    normalized_spa_amplitude_21,
    physical_spa_amplitude_21,
)


MTSUN_SECONDS = 4.925490947641267e-6


def _pn(mode, frequency, mt, m1, m2, chi1, chi2):
    if mode == (2, 1):
        return 2.0 ** (-7.0 / 6.0) * normalized_spa_amplitude_21(
            frequency, mt, m1, m2, chi1, chi2
        )
    if mode in ((3, 2), (4, 4)):
        return normalized_pn_amplitude_compatibility(
            mode, frequency, mt, m1, m2, chi1, chi2
        )
    return normalized_pn_amplitude(mode, frequency, mt, m1, m2, chi1, chi2)


def _synthetic_collocations(mode, f_ins, coeffs, mt, m1, m2, chi1, chi2):
    nodes = torch.tensor(COLLOCATION_FRACTIONS, dtype=torch.float64) * f_ins
    pn = _pn(mode, nodes, mt, m1, m2, chi1, chi2)
    powers = torch.tensor((7.0 / 3.0, 8.0 / 3.0, 3.0), dtype=torch.float64)
    basis = nodes.div(f_ins).unsqueeze(-1).pow(powers)
    return pn + basis @ torch.as_tensor(coeffs, dtype=torch.float64)


def test_independent_three_collocation_values_reconstruct_all_pseudopn_terms():
    mode = (4, 4)
    f_ins = 100.0
    mt = 11.0 * MTSUN_SECONDS
    m1, m2, chi1, chi2 = 6.0, 5.0, 0.12, -0.06
    coefficients = torch.tensor((0.021, -0.013, 0.007), dtype=torch.float64)
    collocations = _synthetic_collocations(
        mode, f_ins, coefficients, mt, m1, m2, chi1, chi2
    )
    frequencies = torch.tensor((35.0, 55.0, 80.0), dtype=torch.float64)
    got = normalized_inspiral_amplitude(
        mode,
        frequencies,
        f_ins,
        collocations,
        mt,
        m1,
        m2,
        chi1,
        chi2,
        apply_vetoes=False,
    )
    pn = _pn(mode, frequencies, mt, m1, m2, chi1, chi2)
    powers = torch.tensor((7.0 / 3.0, 8.0 / 3.0, 3.0), dtype=torch.float64)
    expected = pn + (frequencies / f_ins).unsqueeze(-1).pow(powers) @ coefficients
    torch.testing.assert_close(got, expected, rtol=2e-13, atol=2e-14)

    # Interpolation is exact at the three supplied points by construction.
    at_nodes = normalized_inspiral_amplitude(
        mode,
        torch.tensor(COLLOCATION_FRACTIONS, dtype=torch.float64) * f_ins,
        f_ins,
        collocations,
        mt,
        m1,
        m2,
        chi1,
        chi2,
        apply_vetoes=False,
    )
    torch.testing.assert_close(at_nodes, collocations, rtol=2e-13, atol=2e-14)


def test_printed_vetoes_select_expected_subspaces_without_hidden_coefficients():
    # The normalized values are deliberately nonmonotone, but the supplied
    # geometric amplitudes are monotone; veto selection follows the latter.
    assert printed_veto_nodes(
        (2, 1), 4.0, 0.1, 0.2, (0.10, 0.30, 0.20),
        geometric_veto_amplitudes=(0.30, 0.31, 0.32),
    ) == (0, 1, 2)
    # The 21 threshold is active only for q<8; all nodes are removed here.
    assert printed_veto_nodes(
        (2, 1), 7.9, 0.1, 0.2, (0.10, 0.30, 0.20),
        geometric_veto_amplitudes=(0.10, 0.19, 0.05),
    ) == ()
    # The monotonicity check remains active at q=8 and above.
    assert printed_veto_nodes(
        (2, 1), 8.0, 0.1, 0.2, (0.10, 0.30, 0.20),
        geometric_veto_amplitudes=(0.10, 0.19, 0.05),
    ) == (0, 2)
    with pytest.raises(ValueError, match="geometric_veto_amplitudes"):
        printed_veto_nodes((2, 1), 8.0, 0.1, 0.2, (0.10, 0.30, 0.20))
    with pytest.raises(ValueError, match="geometric_veto_amplitudes"):
        normalized_inspiral_amplitude(
            (2, 1), 20.0, 40.0, (0.10, 0.30, 0.20),
            11.0 * MTSUN_SECONDS, 6.0, 5.0, 0.1, 0.2,
        )
    # Printed 33 special region: remove the last node only.
    assert printed_veto_nodes(
        (3, 3), 1.1, -0.2, 0.1, (0.10, 0.20, 0.30)
    ) == (0, 1)
    assert printed_veto_nodes(
        (3, 3), 1.2, -0.2, 0.1, (0.10, 0.20, 0.30)
    ) == (0, 1, 2)
    assert printed_veto_nodes(
        (4, 4), 3.0, -0.9, 0.9, (0.10, 0.20, 0.30)
    ) == (0, 1, 2)
    # The q<20 portions of the printed 32 veto remove the two highest rows
    # in the documented negative-spin regions; the dynamic ringdown check is
    # deliberately outside this helper's contract.
    assert printed_veto_nodes(
        (3, 2), 3.0, -0.7, 0.2, (0.10, 0.20, 0.30)
    ) == (0,)
    assert printed_veto_nodes(
        (3, 2), 3.0, -0.95, -0.92, (0.10, 0.20, 0.30)
    ) == (0,)


def test_automatic_veto_reconstructs_with_highest_terms_zero_padded():
    mode = (3, 3)
    f_ins = 80.0
    mt = 12.0 * MTSUN_SECONDS
    m1, m2, chi1, chi2 = 5.25, 4.75, -0.2, 0.1
    coefficients = torch.tensor((0.014, -0.008, 0.006), dtype=torch.float64)
    collocations = _synthetic_collocations(
        mode, f_ins, coefficients, mt, m1, m2, chi1, chi2
    )
    # The special 33 veto uses only the first two rows; the third coefficient
    # is consequently zero in the reconstructed ansatz.
    got = normalized_inspiral_amplitude(
        mode,
        torch.tensor((24.0, 40.0), dtype=torch.float64),
        f_ins,
        collocations,
        mt,
        m1,
        m2,
        chi1,
        chi2,
    )
    pn = _pn(
        mode, torch.tensor((24.0, 40.0), dtype=torch.float64), mt, m1, m2, chi1, chi2
    )
    retained = torch.tensor((0.5, 0.75), dtype=torch.float64)
    # Solve the independent two-row system for alpha and beta.
    nodes = torch.tensor((0.5, 0.75, 1.0), dtype=torch.float64)
    pn_nodes = _pn(mode, nodes * f_ins, mt, m1, m2, chi1, chi2)
    matrix = retained.unsqueeze(-1).pow(
        torch.tensor((7.0 / 3.0, 8.0 / 3.0), dtype=torch.float64)
    )
    rhs = collocations[:2] - pn_nodes[:2]
    expected_ab = torch.linalg.solve(matrix, rhs)
    probe = torch.tensor((24.0, 40.0), dtype=torch.float64)
    probe_basis = (probe / f_ins).unsqueeze(-1).pow(
        torch.tensor((7.0 / 3.0, 8.0 / 3.0), dtype=torch.float64)
    )
    expected = pn + probe_basis @ expected_ab
    torch.testing.assert_close(got, expected, rtol=2e-12, atol=2e-13)

    # Canonicalizing the heavier body as body 1 makes the asymmetric printed
    # spin-region veto invariant under an exchanged input ordering.
    exchanged = normalized_inspiral_amplitude(
        mode,
        torch.tensor((24.0, 40.0), dtype=torch.float64),
        f_ins,
        collocations,
        mt,
        m2,
        m1,
        chi2,
        chi1,
    )
    torch.testing.assert_close(exchanged, got, rtol=2e-12, atol=2e-13)


def test_batch_scalar_exchange_and_spin_gradients():
    mode = (4, 4)
    mt = torch.tensor((10.0, 13.0), dtype=torch.float64) * MTSUN_SECONDS
    m1 = torch.tensor((6.0, 8.0), dtype=torch.float64)
    m2 = torch.tensor((4.0, 5.0), dtype=torch.float64)
    chi1 = torch.tensor((0.12, -0.2), dtype=torch.float64)
    chi2 = torch.tensor((-0.04, 0.1), dtype=torch.float64)
    f_ins = torch.tensor((90.0, 110.0), dtype=torch.float64)
    frequency = torch.tensor((30.0, 55.0), dtype=torch.float64)
    collocations = torch.stack(
        tuple(
            _synthetic_collocations(
                mode,
                f_ins[index],
                (0.011, -0.004, 0.002),
                mt[index],
                m1[index],
                m2[index],
                chi1[index],
                chi2[index],
            )
            for index in range(2)
        )
    )
    batch = normalized_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        collocations,
        mt,
        m1,
        m2,
        chi1,
        chi2,
        retained_nodes=(0, 1, 2),
    )
    assert batch.shape == (2,)
    for index in range(2):
        scalar = normalized_inspiral_amplitude(
            mode,
            frequency[index],
            f_ins[index],
            collocations[index],
            mt[index],
            m1[index],
            m2[index],
            chi1[index],
            chi2[index],
            apply_vetoes=False,
        )
        torch.testing.assert_close(batch[index], scalar, rtol=1e-13, atol=1e-14)

    exchanged = normalized_inspiral_amplitude(
        mode,
        frequency[0],
        f_ins[0],
        collocations[0],
        mt[0],
        m2[0],
        m1[0],
        chi2[0],
        chi1[0],
        apply_vetoes=False,
    )
    torch.testing.assert_close(exchanged, batch[0], rtol=2e-13, atol=2e-14)

    # The real amplitude remains differentiable through both the frequency
    # and complex-valued 21 PN baseline's spin-sensitive magnitude.
    f = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
    s2 = torch.tensor(-0.07, dtype=torch.float64, requires_grad=True)
    value = normalized_inspiral_amplitude(
        (2, 1), f, 90.0, (0.08, 0.10, 0.12), 11.0 * MTSUN_SECONDS,
        6.0, 5.0, s1, s2, apply_vetoes=False,
    )
    gradients = torch.autograd.grad(value, (f, s1, s2))
    assert all(torch.isfinite(item) for item in gradients)
    assert all(float(item.abs()) > 1e-10 for item in gradients)

    def scalar_value(freq, spin1, spin2):
        return float(normalized_inspiral_amplitude(
            (2, 1), freq, 90.0, (0.08, 0.10, 0.12), 11.0 * MTSUN_SECONDS,
            6.0, 5.0, spin1, spin2, apply_vetoes=False,
        ))

    h = 1e-4
    finite_differences = (
        (scalar_value(35.0 + h, 0.13, -0.07) - scalar_value(35.0 - h, 0.13, -0.07)) / (2 * h),
        (scalar_value(35.0, 0.13 + h, -0.07) - scalar_value(35.0, 0.13 - h, -0.07)) / (2 * h),
        (scalar_value(35.0, 0.13, -0.07 + h) - scalar_value(35.0, 0.13, -0.07 - h)) / (2 * h),
    )
    torch.testing.assert_close(
        torch.stack(gradients), torch.tensor(finite_differences, dtype=torch.float64),
        rtol=2e-6, atol=2e-8
    )


def test_physical_scale_and_printed_cut_formula():
    f_ins = 0.03
    expected_21 = 0.5 * (0.02 + 0.01 * (0.75 - 0.235 * 0.2 - 5.0 / 6.0 * 0.2**2))
    expected_33 = 1.5 * (0.02 + 0.01 * (0.75 - 0.235 * 0.2 - 5.0 / 6.0 * 0.2))
    expected_44 = 2.0 * (0.02 + 0.01 * (0.75 - 0.235 * 0.2))
    expected_32 = 1.0 * (
        0.02 * (0.04 / 0.03) + 0.01 * (0.75 - 0.235 * abs(0.2))
    )
    torch.testing.assert_close(
        printed_inspiral_cut_frequency((2, 1), 0.02, 0.03, 0.2),
        torch.tensor(expected_21, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        printed_inspiral_cut_frequency((3, 3), 0.02, 0.03, 0.2),
        torch.tensor(expected_33, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        printed_inspiral_cut_frequency((4, 4), 0.02, 0.03, 0.2),
        torch.tensor(expected_44, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        printed_inspiral_cut_frequency(
            (3, 2), 0.02, 0.03, -0.2, f_rd_32=0.04, f_rd_22=0.03
        ),
        torch.tensor(expected_32, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    with pytest.raises(ValueError, match="f_rd_32"):
        printed_inspiral_cut_frequency((3, 2), 0.02, 0.03, 0.2)


def test_32_release_compatibility_cut_scales_the_complete_expression():
    meco, isco, spin = 0.02, 0.03, -0.2
    f_rd_32, f_rd_22 = 0.04, 0.03
    epsilon = f_rd_32 / f_rd_22
    delta = 3.0 / 4.0 - 0.235 * abs(spin)
    printed = (2.0 / 2.0) * (epsilon * meco + abs(isco - meco) * delta)
    compatibility = (2.0 / 2.0) * epsilon * (
        meco + abs(isco - meco) * delta
    )

    # Keep this equation-level check independent of the helper implementation.
    torch.testing.assert_close(
        printed_inspiral_cut_frequency(
            (3, 2), meco, isco, spin, f_rd_32=f_rd_32, f_rd_22=f_rd_22
        ),
        torch.tensor(printed, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        compatibility_inspiral_cut_frequency(
            (3, 2), meco, isco, spin, f_rd_32=f_rd_32, f_rd_22=f_rd_22
        ),
        torch.tensor(compatibility, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        inspiral_cut_frequency(
            (3, 2), meco, isco, spin, f_rd_32=f_rd_32, f_rd_22=f_rd_22,
            version=COMPATIBILITY_CUT_VERSION,
        ),
        torch.tensor(compatibility, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    torch.testing.assert_close(
        inspiral_cut_frequency(
            (3, 2), meco, isco, spin, f_rd_32=f_rd_32, f_rd_22=f_rd_22,
            version=PRINTED_CUT_VERSION,
        ),
        torch.tensor(printed, dtype=torch.float64), rtol=0.0, atol=1e-15,
    )
    assert compatibility != printed


def test_diagnostic_frequency_and_veto_support_checks_are_explicit():
    frequencies = frequencies_below(100.0)
    assert frequencies.tolist() == [50.0, 75.0, 90.0, 95.0]
    assert validate_frequency_support(frequencies, 100.0).tolist() == frequencies.tolist()
    assert validate_frequency_support((50.0, 75.0, 100.0), 100.0, allow_endpoint=True).tolist() == [50.0, 75.0, 100.0]
    with pytest.raises(ValueError, match="strictly below"):
        validate_frequency_support((50.0, 100.0), 100.0)
    with pytest.raises(ValueError, match="at or below"):
        validate_frequency_support((50.0, 100.1), 100.0, allow_endpoint=True)

    assert validate_veto_support((0, 2)) == (0, 2)
    with pytest.raises(ValueError, match="at least one"):
        validate_veto_support(())
    with pytest.raises(ValueError, match="sorted"):
        validate_veto_support((2, 0))
    with pytest.raises(ValueError, match="outside"):
        validate_veto_support((0, 3))

    common = dict(
        mode=(4, 4), frequency=30.0, f_ins=100.0,
        collocation_values=(0.1, 0.12, 0.14),
        m1=6.0, m2=4.0, chi1=0.08, chi2=0.03, apply_vetoes=False,
    )
    distance = 500e6 * 3.0856775814913673e16 / 299792458.0
    first = physical_inspiral_amplitude(
        total_mass_seconds=10.0 * MTSUN_SECONDS, distance_seconds=distance, **common
    )
    doubled = physical_inspiral_amplitude(
        total_mass_seconds=20.0 * MTSUN_SECONDS, distance_seconds=distance,
        frequency=15.0, f_ins=50.0, m1=12.0, m2=8.0,
        mode=(4, 4), collocation_values=common["collocation_values"],
        chi1=0.08, chi2=0.03, apply_vetoes=False,
    )
    torch.testing.assert_close(
        doubled / first, torch.tensor(4.0, dtype=torch.float64), rtol=2e-13, atol=2e-14
    )
    assert math.isfinite(float(first)) and float(first) > 0


def test_21_physical_assembly_matches_nonreexpanded_spa_prefactor():
    frequency = torch.tensor((18.0, 32.0, 48.0), dtype=torch.float64)
    total_mass_seconds = 11.0 * MTSUN_SECONDS
    m1, m2, chi1, chi2 = 6.0, 5.0, 0.13, -0.07
    distance = 500e6 * 3.0856775814913673e16 / 299792458.0
    f_ins = 80.0
    assembled = physical_inspiral_amplitude(
        (2, 1),
        frequency,
        f_ins,
        (0.08, 0.10, 0.12),
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        distance,
        retained_nodes=(),
        apply_vetoes=False,
    )
    expected = physical_spa_amplitude_21(
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        distance,
    )
    torch.testing.assert_close(assembled, expected, rtol=3e-13, atol=3e-25)


def test_domain_checks_and_explicit_units():
    args = dict(
        mode=(4, 4), frequency=20.0, f_ins=30.0,
        collocation_values=(0.1, 0.12, 0.14),
        total_mass_seconds=10.0 * MTSUN_SECONDS,
        m1=6.0, m2=4.0, chi1=0.1, chi2=0.0, apply_vetoes=False,
    )
    assert torch.isfinite(normalized_inspiral_amplitude(**args))
    with pytest.raises(ValueError, match="supported clean"):
        normalized_inspiral_amplitude(**{**args, "mode": (5, 5)})
    with pytest.raises(ValueError, match="q < 20"):
        normalized_inspiral_amplitude(**{**args, "m1": 80.0, "m2": 2.0})
    with pytest.raises(ValueError, match="frequency <= f_ins"):
        normalized_inspiral_amplitude(**{**args, "frequency": 40.0})
    with pytest.raises(ValueError, match="exactly three"):
        normalized_inspiral_amplitude(**{**args, "collocation_values": (0.1, 0.2)})
    with pytest.raises(ValueError, match="distance_seconds"):
        physical_inspiral_amplitude(**args, distance_seconds=0.0)


def test_diagnostic_uses_xas_final_spin_for_kerr_isco():
    cases = (
        (40.0, 15.0, 0.3, -0.2, 0.6792822582610814, 0.04427066667167863),
        (55.0, 11.0, -0.4, 0.2, 0.1801364947977045, 0.02502280544699784),
        (27.0, 16.0, -0.25, 0.15, 0.5924309094209793, 0.03892305405670921),
        (35.0, 20.0, 0.2, 0.1, 0.7073333174012265, 0.04636758312030512),
    )
    for m1, m2, chi1, chi2, expected_af, expected_fisco in cases:
        eta = m1 * m2 / (m1 + m2) ** 2
        final_spin, fisco = final_spin_and_f_isco(eta, chi1, chi2)
        assert final_spin == pytest.approx(expected_af, rel=2e-13, abs=2e-15)
        assert fisco == pytest.approx(expected_fisco, rel=2e-13, abs=2e-15)


def test_diagnostic_chi_pn_contract_is_published_eq18():
    eta = 40.0 * 15.0 / 55.0**2
    chi_eff = (40.0 * 0.3 + 15.0 * -0.2) / 55.0
    expected = chi_eff - 38.0 * eta * (0.3 - 0.2) / 113.0
    assert chi_pn(eta, chi_eff, 0.3, -0.2) == pytest.approx(expected)


def test_provider_inspiral_input_and_order_are_explicit(monkeypatch):
    calls = []

    def fake_spin_variables(eta, spin1, spin2):
        return tuple(torch.tensor(value, dtype=torch.float64) for value in (0.1, 0.2, 0.3, 0.4))

    monkeypatch.setattr(
        XHMParameterFits,
        "compute_spin_variables",
        staticmethod(fake_spin_variables),
    )
    for index in (1, 2, 3):
        def fake_fit(eta, spin1, spin2, spin_input, index=index):
            calls.append(float(spin_input))
            return torch.tensor(float(index) * (-1.0 if index % 2 else 1.0), dtype=torch.float64)

        monkeypatch.setattr(
            XHMParameterFits,
            f"amp_insp{index}_21",
            staticmethod(fake_fit),
        )

    result = provider_collocations(40.0, 15.0, 0.3, -0.2, (2, 1))
    eta = 40.0 * 15.0 / 55.0**2
    chi_eff = (40.0 * 0.3 + 15.0 * -0.2) / 55.0
    expected_chi_pn = chi_pn(eta, chi_eff, 0.3, -0.2)
    expected_chi_pn_hat = expected_chi_pn / (1.0 - 76.0 * eta / 113.0)
    assert result.raw_amp_insp123.tolist() == [-1.0, 2.0, -3.0]
    assert result.ordered_h_lm.tolist() == [3.0, 2.0, 1.0]
    assert calls == pytest.approx([expected_chi_pn_hat] * 3)
    assert result.chi_pn_raw == pytest.approx(expected_chi_pn)
    assert result.chi_pn == pytest.approx(expected_chi_pn_hat)
    assert result.shat == pytest.approx(0.4)


def test_common_h_geometric_conversion_uses_published_a22_zero():
    h_lm = torch.tensor((0.1, 0.2, 0.3), dtype=torch.float64).numpy()
    frequencies = torch.tensor((0.01, 0.02, 0.03), dtype=torch.float64).numpy()
    mass_seconds = 55.0 * MTSUN_SECONDS
    eta = 40.0 * 15.0 / 55.0**2
    expected = h_lm * math.pi * math.sqrt(2.0 * eta / 3.0) * (
        math.pi * frequencies * mass_seconds
    ) ** (-7.0 / 6.0)
    got = geometric_amplitudes_from_common_h(h_lm, frequencies, mass_seconds, eta)
    assert got == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_diagnostic_module_collects_without_lal():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root / "src"), str(repo_root / "tests"), str(repo_root))
    )
    code = r'''
import builtins

real_import = builtins.__import__

def blocked_lal(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"lal", "lalsimulation"}:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_lal
import scripts.diagnose_xhm_inspiral_assembly_clean as diagnostic
assert diagnostic.DISTANCE_MPC == 500.0
assert diagnostic.DISTANCE_M is None
assert diagnostic.DISTANCE_SECONDS is None
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
