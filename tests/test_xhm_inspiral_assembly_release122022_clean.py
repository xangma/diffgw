# SPDX-License-Identifier: MIT
"""Contract tests for the clean XHM release-122022 inspiral adapter.

The node values in these tests are independent synthetic geometric amplitudes;
they are algebraic fixtures, not values obtained from a waveform oracle or a
runtime fit provider.  These tests validate the assembly contract only.
"""
from __future__ import annotations

import math

import pytest
import torch

from torchwave.approximants.xhm_amplitude_clean import (
    hhat_lm_compatibility,
    hhat_lm_printed,
)
from torchwave.approximants.xhm_inspiral_assembly_release122022_clean import (
    CALIBRATION_QUANTITY,
    COLLOCATION_FRACTIONS,
    DEFAULT_RETAINED_NODES,
    PSEUDO_PN_POWERS,
    RELEASE_VERSION,
    SUPPORTED_MODES,
    CalibrationProvenance,
    geometric_inspiral_amplitude,
    inspiral_cut_frequency,
    normalized_inspiral_amplitude,
    physical_inspiral_amplitude,
    pn_geometric_amplitude,
    pn_normalized_amplitude,
    validate_calibration_provenance,
)


PROVENANCE = CalibrationProvenance(
    source="independent synthetic collocation fixture",
)
DTYPE = torch.float64


def _case(mode=(3, 3)):
    # The small total mass keeps the dimensionless PN velocity in the range
    # where the printed Appendix-E polynomials are a useful synthetic base.
    total_mass = 1.1e-3
    mass1, mass2 = 0.7, 0.4
    chi1, chi2 = 0.23, -0.17
    f_meco = 140.0
    f_ins = inspiral_cut_frequency(mode, f_meco)
    return total_mass, mass1, mass2, chi1, chi2, f_meco, f_ins


def _geometric_scale(frequency, total_mass, mass1, mass2):
    frequency = torch.as_tensor(frequency, dtype=DTYPE)
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    return (
        math.pi
        * torch.sqrt(torch.as_tensor(2.0 * eta / 3.0, dtype=DTYPE))
        * (math.pi * frequency * total_mass) ** (-7.0 / 6.0)
    )


def _synthetic_nodes(mode, coefficients=(0.004, -0.002, 0.001)):
    total_mass, mass1, mass2, chi1, chi2, _, f_ins = _case(mode)
    node_fraction = torch.tensor(COLLOCATION_FRACTIONS, dtype=DTYPE)
    node_frequency = f_ins * node_fraction
    coefficients = torch.as_tensor(coefficients, dtype=DTYPE)
    correction = (node_fraction[:, None] ** torch.tensor(
        PSEUDO_PN_POWERS, dtype=DTYPE
    ) * coefficients).sum(dim=-1)
    baseline = pn_geometric_amplitude(
        mode,
        node_frequency,
        total_mass,
        mass1,
        mass2,
        chi1,
        chi2,
    )
    values = baseline + _geometric_scale(
        node_frequency, total_mass, mass1, mass2
    ) * correction
    assert torch.isfinite(values).all()
    assert (values > 0).all()
    return (
        values,
        coefficients,
        total_mass,
        mass1,
        mass2,
        chi1,
        chi2,
        f_ins,
    )


def _expected_geometric(mode, frequency, coefficients):
    total_mass, mass1, mass2, chi1, chi2, _, f_ins = _case(mode)
    frequency = torch.as_tensor(frequency, dtype=DTYPE)
    u = frequency / f_ins
    correction = (
        u[..., None] ** torch.tensor(PSEUDO_PN_POWERS, dtype=DTYPE)
        * coefficients
    ).sum(dim=-1)
    normalized = pn_normalized_amplitude(
        mode, frequency, total_mass, mass1, mass2, chi1, chi2
    ) + correction
    return _geometric_scale(frequency, total_mass, mass1, mass2) * normalized


def test_current_release_cut_is_mode_scaled_meco_and_domain_is_explicit():
    meco = torch.tensor((100.0, 140.0), dtype=DTYPE)
    for mode in SUPPORTED_MODES:
        expected = mode[1] / 2.0 * meco
        torch.testing.assert_close(inspiral_cut_frequency(mode, meco), expected)

    with pytest.raises(ValueError, match="supports only modes"):
        inspiral_cut_frequency((2, 2), 100.0)
    with pytest.raises(ValueError, match="positive"):
        inspiral_cut_frequency((3, 3), 0.0)


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_reexpanded_pn_baseline_and_three_node_recovery(mode):
    values, coefficients, total_mass, m1, m2, chi1, chi2, f_ins = _synthetic_nodes(
        mode
    )
    frequencies = f_ins * torch.tensor((0.31, 0.62, 0.91), dtype=DTYPE)
    expected = _expected_geometric(mode, frequencies, coefficients)
    recovered = geometric_inspiral_amplitude(
        mode,
        frequencies,
        f_ins,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        calibration_provenance=PROVENANCE,
    )
    torch.testing.assert_close(recovered, expected, rtol=2e-12, atol=2e-12)

    node_recovery = geometric_inspiral_amplitude(
        mode,
        f_ins * torch.tensor(COLLOCATION_FRACTIONS, dtype=DTYPE),
        f_ins,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        calibration_provenance=PROVENANCE,
    )
    torch.testing.assert_close(node_recovery, values, rtol=2e-12, atol=2e-12)


def test_21_uses_reexpanded_appendix_e_not_legacy_spa_branch():
    mode = (2, 1)
    frequency = torch.tensor((16.0, 43.0), dtype=DTYPE)
    total_mass, m1, m2, chi1, chi2, _, _ = _case(mode)
    eta = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    v = (2.0 * math.pi * frequency * total_mass / mode[1]) ** (1.0 / 3.0)
    expected = hhat_lm_printed(
        mode, v, eta, delta, chi_s, chi_a
    ).abs() * (mode[1] / 2.0) ** (7.0 / 6.0)
    actual = pn_normalized_amplitude(
        mode, frequency, total_mass, m1, m2, chi1, chi2
    )
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_named_32_44_compatibility_signs_are_retained():
    for mode in ((3, 2), (4, 4)):
        total_mass, m1, m2, chi1, chi2, _, _ = _case(mode)
        frequency = torch.tensor((25.0, 70.0), dtype=DTYPE)
        v = (2.0 * math.pi * frequency * total_mass / mode[1]) ** (1.0 / 3.0)
        eta = m1 * m2 / (m1 + m2) ** 2
        delta = (m1 - m2) / (m1 + m2)
        chi_s = (chi1 + chi2) / 2.0
        chi_a = (chi1 - chi2) / 2.0
        expected = hhat_lm_compatibility(
            mode, v, eta, delta, chi_s, chi_a
        ).abs() * (mode[1] / 2.0) ** (7.0 / 6.0)
        actual = pn_normalized_amplitude(
            mode, frequency, total_mass, m1, m2, chi1, chi2
        )
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_retained_node_tuple_is_explicit_and_row_permutation_is_algebraic():
    mode = (3, 3)
    values, coefficients, total_mass, m1, m2, chi1, chi2, f_ins = _synthetic_nodes(
        mode, coefficients=(0.006, -0.003, 0.0)
    )
    frequency = 0.63 * f_ins
    expected = _expected_geometric(mode, frequency, coefficients)
    recovered = geometric_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        retained_nodes=(2, 0),
        calibration_provenance=PROVENANCE,
    )
    torch.testing.assert_close(recovered, expected, rtol=2e-12, atol=2e-12)

    pn_only = geometric_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        retained_nodes=(),
        calibration_provenance=PROVENANCE,
    )
    torch.testing.assert_close(
        pn_only,
        pn_geometric_amplitude(
            mode, frequency, total_mass, m1, m2, chi1, chi2
        ),
        rtol=2e-14,
        atol=2e-14,
    )

    # The three trailing values have a fixed ascending node order; the
    # adapter does not silently reorder an explicitly supplied array.
    reversed_values = values.flip(-1)
    reversed_result = geometric_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        reversed_values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        calibration_provenance=PROVENANCE,
    )
    assert not torch.allclose(reversed_result, expected, rtol=1e-8, atol=1e-10)


def test_broadcasting_scalar_calls_physical_normalization_and_distance():
    mode = (4, 4)
    values0, coefficients0, total_mass0, m10, m20, s10, s20, cut0 = _synthetic_nodes(
        mode
    )
    values1, coefficients1, total_mass1, m11, m21, s11, s21, cut1 = _synthetic_nodes(
        mode, coefficients=(0.002, -0.001, 0.0003)
    )
    del coefficients0, coefficients1
    frequencies = torch.tensor((0.42 * cut0, 0.58 * cut1), dtype=DTYPE)
    cuts = torch.tensor((cut0, cut1), dtype=DTYPE)
    values = torch.stack((values0, values1))
    total_mass = torch.tensor((total_mass0, total_mass1), dtype=DTYPE)
    m1 = torch.tensor((m10, m11), dtype=DTYPE)
    m2 = torch.tensor((m20, m21), dtype=DTYPE)
    chi1 = torch.tensor((s10, s11), dtype=DTYPE)
    chi2 = torch.tensor((s20, s21), dtype=DTYPE)
    batched = normalized_inspiral_amplitude(
        mode,
        frequencies,
        cuts,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        calibration_provenance=PROVENANCE,
    )
    for index in range(2):
        scalar = normalized_inspiral_amplitude(
            mode,
            frequencies[index],
            cuts[index],
            values[index],
            total_mass[index],
            m1[index],
            m2[index],
            chi1[index],
            chi2[index],
            calibration_provenance=PROVENANCE,
        )
        torch.testing.assert_close(batched[index], scalar, rtol=2e-13, atol=2e-13)

    distance = torch.tensor((2.0, 5.0), dtype=DTYPE)
    physical = physical_inspiral_amplitude(
        mode,
        frequencies,
        cuts,
        values,
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        distance,
        calibration_provenance=PROVENANCE,
    )
    expected_physical = (
        total_mass.square() / distance
        * _geometric_scale(frequencies, total_mass, m1, m2)
        * batched
    )
    torch.testing.assert_close(physical, expected_physical, rtol=2e-13, atol=2e-13)


def test_exchange_float32_and_gradients_are_finite():
    mode = (2, 1)
    values, _, _, _, _, _, _, cut = _synthetic_nodes(mode)
    frequency = torch.tensor(
        (0.34 * cut, 0.72 * cut), dtype=torch.float32, requires_grad=True
    )
    f_ins = torch.as_tensor(cut, dtype=torch.float32).detach().requires_grad_()
    total_mass = torch.tensor(1.1e-3, dtype=torch.float32, requires_grad=True)
    m1 = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
    m2 = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
    chi1 = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    chi2 = torch.tensor(-0.17, dtype=torch.float32, requires_grad=True)
    result = geometric_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        values.to(torch.float32),
        total_mass,
        m1,
        m2,
        chi1,
        chi2,
        calibration_provenance=PROVENANCE,
    )
    assert result.dtype == DTYPE
    gradients = torch.autograd.grad(
        result.sum(),
        (frequency, f_ins, total_mass, m1, m2, chi1, chi2),
        allow_unused=False,
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    exchanged = geometric_inspiral_amplitude(
        mode,
        torch.tensor((0.34 * cut, 0.72 * cut), dtype=DTYPE),
        cut,
        values,
        1.1e-3,
        0.4,
        0.7,
        -0.17,
        0.23,
        calibration_provenance=PROVENANCE,
    )
    torch.testing.assert_close(result.detach(), exchanged, rtol=4e-6, atol=4e-8)


def test_provenance_and_boundary_policy_reject_ambiguous_inputs():
    mode = (3, 3)
    values, _, total_mass, m1, m2, chi1, chi2, cut = _synthetic_nodes(mode)
    common = dict(
        mode=mode,
        frequency=0.4 * cut,
        f_ins=cut,
        collocation_values=values,
        total_mass_seconds=total_mass,
        m1=m1,
        m2=m2,
        chi1=chi1,
        chi2=chi2,
    )
    with pytest.raises(TypeError, match="calibration_provenance"):
        normalized_inspiral_amplitude(**common)
    with pytest.raises(ValueError, match="release"):
        normalized_inspiral_amplitude(
            **common,
            calibration_provenance={
                "source": "old normalized fixture",
                "release": 122019,
                "quantity": CALIBRATION_QUANTITY,
                "node_fractions": COLLOCATION_FRACTIONS,
            },
        )
    with pytest.raises(ValueError, match="quantity"):
        validate_calibration_provenance(
            {
                "source": "wrong units",
                "release": RELEASE_VERSION,
                "quantity": "normalized_H",
                "node_fractions": COLLOCATION_FRACTIONS,
            }
        )
    with pytest.raises(ValueError, match="node_fractions"):
        validate_calibration_provenance(
            CalibrationProvenance(
                "wrong node order", node_fractions=(1.0, 0.75, 0.5)
            )
        )
    with pytest.raises(ValueError, match="frequency <= f_ins"):
        normalized_inspiral_amplitude(
            **{**common, "frequency": 1.1 * cut},
            calibration_provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="q < 20"):
        normalized_inspiral_amplitude(
            **{**common, "m1": 20.0, "m2": 1.0},
            calibration_provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="distinct"):
        normalized_inspiral_amplitude(
            **common,
            retained_nodes=(0, 0),
            calibration_provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="0, 1, or 2"):
        normalized_inspiral_amplitude(
            **common,
            retained_nodes=(0, 3),
            calibration_provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="trailing node"):
        normalized_inspiral_amplitude(
            **{**common, "collocation_values": (0.1, 0.2)},
            calibration_provenance=PROVENANCE,
        )
