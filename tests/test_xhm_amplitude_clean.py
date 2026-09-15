"""Clean-room numeric checks for the Appendix-E Fourier mode amplitudes.

The coefficient table is an independent high-precision transcription of the
five displayed Appendix-E brackets, including each mode's prefactor.  The
implementation is evaluated at several v values and the coefficients are
also recovered from those values, so these checks do not repeat its formula.
"""

import math

import pytest
import torch

from torchwave.approximants.xhm_amplitude_clean import (
    collocation_coefficients,
    hhat_lm,
    hhat_lm_compatibility,
    hhat_lm_printed,
    normalized_pn_amplitude,
)
from torchwave.approximants.xhm_spa_clean import spa_hhat_21


# eta=.21 and delta=.4 are physically consistent: delta**2 == 1 - 4*eta.
# The spins correspond to chi_1=.17 and chi_2=.09.
_ETA, _DELTA, _CHI_S, _CHI_A = 0.21, 0.4, 0.13, 0.04
_MODES = ((2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
_V_NODES = torch.tensor((0.20, 0.35, 0.50, 0.65, 0.80, 0.95), dtype=torch.float64)


# Final Hhat_lm coefficient of each power v^1,...,v^6.  Zero entries are
# intentional: they record powers absent from the corresponding published
# bracket.  Values are fixed checkpoints from a separate transcription, not
# expressions evaluated from the implementation under test.
_EXPECTED_COEFFICIENTS = {
    (2, 1): (
        0.0 + 0.18856180831641267j,
        0.0 - 0.06505382386916237j,
        0.0 + 0.17673179962799012j,
        0.35568307574981892 - 0.52134429166709844j,
        0.0 + 0.24390043103141973j,
        0.79099622094570042 - 0.011295109361428364j,
    ),
    (3, 3): (
        0.0 - 0.25354627641855497j,
        0.0 + 0.0j,
        0.0 + 0.55414953437717271j,
        -0.44806935068709936 - 0.90016770828511640j,
        0.0 + 1.2563483500800460j,
        -0.74622153902937024 + 1.7117545648253082j,
    ),
    (3, 2): (
        0.0 + 0.0j,
        0.10423569141651704 + 0.0j,
        0.030763614872118003 + 0.0j,
        -0.032594223459575604 + 0.0j,
        -0.10553910708936502 + 0.06423172335936726j,
        0.36388280324628402 + 0.0j,
    ),
    (4, 4): (
        0.0 + 0.0j,
        -0.19654870464610029 + 0.0j,
        0.0 + 0.0j,
        0.62434505497828438 + 0.0j,
        1.3530711530463755 - 0.045171325585998460j,
        -1.0317673994978971 + 0.0j,
    ),
    (4, 3): (
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 - 0.050941983807913432j,
        0.0 - 0.0013833383534045458j,
        0.0 + 0.07870374904479377j,
        -0.30925509665533022 - 0.19258232987997287j,
    ),
}


def _checkpoint_coefficients(mode):
    return torch.tensor(_EXPECTED_COEFFICIENTS[mode], dtype=torch.complex128)


def _checkpoint_values(mode):
    coefficients = _checkpoint_coefficients(mode)
    powers = torch.arange(1, 7, dtype=torch.float64)
    design = _V_NODES[:, None].pow(powers)
    expected = design.to(torch.complex128) @ coefficients
    got = torch.stack(
        [hhat_lm(mode, v, _ETA, _DELTA, _CHI_S, _CHI_A) for v in _V_NODES]
    )
    return design, coefficients, expected, got


@pytest.mark.parametrize("mode", _MODES)
def test_appendix_e10_e11_e12_e14_e15_per_order_checkpoints(mode):
    """Check every displayed v^1 through v^6 coefficient independently."""
    assert math.isclose(_DELTA**2, 1.0 - 4.0 * _ETA, rel_tol=0.0, abs_tol=1e-15)
    design, expected_coefficients, expected, got = _checkpoint_values(mode)

    # Direct values are evaluated independently at six v points.
    torch.testing.assert_close(got, expected, rtol=3e-13, atol=3e-15)

    # Recover every power coefficient from the six values, including the
    # structural zero orders, to keep a single mistaken term from hiding in a
    # sum at one v value.
    recovered = torch.linalg.solve(design.to(torch.complex128), got)
    torch.testing.assert_close(recovered, expected_coefficients, rtol=5e-10, atol=3e-12)


def test_22_is_only_the_leading_reference_normalization():
    values = hhat_lm(
        (2, 2), _V_NODES, _ETA, _DELTA, _CHI_S, _CHI_A
    )
    torch.testing.assert_close(values, torch.ones_like(values))


def test_named_compatibility_branch_keeps_printed_paper_available():
    values = torch.tensor((0.11, 0.19), dtype=torch.float64)
    for mode in ((3, 2), (4, 4)):
        printed = hhat_lm_printed(mode, values, _ETA, _DELTA, _CHI_S, _CHI_A)
        compatibility = hhat_lm_compatibility(
            mode, values, _ETA, _DELTA, _CHI_S, _CHI_A
        )
        torch.testing.assert_close(
            hhat_lm(mode, values, _ETA, _DELTA, _CHI_S, _CHI_A),
            printed,
        )
        torch.testing.assert_close(
            hhat_lm(
                mode,
                values,
                _ETA,
                _DELTA,
                _CHI_S,
                _CHI_A,
                version="compatibility",
            ),
            compatibility,
        )
        assert not torch.allclose(printed, compatibility)


def test_odd_modes_vanish_at_equal_mass_and_float32_inputs_widen_cleanly():
    for mode in ((2, 1), (3, 3), (4, 3)):
        value = hhat_lm(mode, 0.117, 0.25, 0.0, 0.13, 0.0)
        torch.testing.assert_close(value, torch.zeros_like(value))
    equal_spa = spa_hhat_21(0.117, 0.25, 0.0, 0.13, 0.0)
    torch.testing.assert_close(equal_spa, torch.zeros_like(equal_spa))
    spa = normalized_pn_amplitude(
        (3, 2),
        torch.tensor((20.0, 40.0), dtype=torch.float32),
        55.0 * 4.925490947641267e-6,
        40.0,
        15.0,
        0.3,
        -0.2,
    )
    assert spa.dtype == torch.float64
    assert torch.isfinite(spa).all()


def test_exchange_symmetry_for_all_published_modes():
    odd_modes = ((2, 1), (3, 3), (4, 3))
    even_modes = ((3, 2), (4, 4), (2, 2))
    for mode in odd_modes:
        original = hhat_lm(mode, 0.117, _ETA, _DELTA, _CHI_S, _CHI_A)
        exchanged = hhat_lm(mode, 0.117, _ETA, -_DELTA, _CHI_S, -_CHI_A)
        torch.testing.assert_close(original, -exchanged, rtol=2e-13, atol=2e-15)
    for mode in even_modes:
        original = hhat_lm(mode, 0.117, _ETA, _DELTA, _CHI_S, _CHI_A)
        exchanged = hhat_lm(mode, 0.117, _ETA, -_DELTA, _CHI_S, -_CHI_A)
        torch.testing.assert_close(original, exchanged, rtol=2e-13, atol=2e-15)


def _finite_difference(mode, v, args, index, component, step=1e-6):
    high = list(args)
    high[index] += step
    low = list(args)
    low[index] -= step
    high_value = hhat_lm(mode, v, *high)
    low_value = hhat_lm(mode, v, *low)
    high_component = high_value.real if component == "real" else high_value.imag
    low_component = low_value.real if component == "real" else low_value.imag
    return (high_component - low_component) / (2.0 * step)


@pytest.mark.parametrize("mode", _MODES)
def test_phase_sensitive_real_and_imag_gradients(mode):
    v = torch.tensor(0.117, dtype=torch.float64)
    eta = torch.tensor(_ETA, dtype=torch.float64, requires_grad=True)
    chi_s = torch.tensor(_CHI_S, dtype=torch.float64, requires_grad=True)
    chi_a = torch.tensor(_CHI_A, dtype=torch.float64, requires_grad=True)
    value = hhat_lm(mode, v, eta, _DELTA, chi_s, chi_a)
    gradients = {}
    for component, scalar in (("real", value.real), ("imag", value.imag)):
        gradients[component] = torch.autograd.grad(
            scalar, (eta, chi_s, chi_a), retain_graph=component == "real"
        )
        vector_norm = torch.linalg.vector_norm(torch.stack(gradients[component]))
        assert torch.isfinite(vector_norm)
        assert float(vector_norm) > 1e-12

    args = [_ETA, _DELTA, _CHI_S, _CHI_A]
    for component in ("real", "imag"):
        for gradient, index in zip(gradients[component], (0, 2, 3)):
            finite_difference = _finite_difference(mode, float(v), args, index, component)
            assert torch.isfinite(gradient)
            torch.testing.assert_close(gradient, finite_difference, rtol=5e-6, atol=5e-10)


def test_normalized_amplitude_is_finite_and_frequency_differentiable():
    frequency = torch.tensor((20.0, 100.0), dtype=torch.float64, requires_grad=True)
    amplitude = normalized_pn_amplitude(
        (3, 2), frequency, 55.0 * 4.925490947641267e-6, 40.0, 15.0, 0.3, -0.2
    )
    assert torch.isfinite(amplitude).all()
    derivative = torch.autograd.grad(amplitude.sum(), frequency)[0]
    assert torch.isfinite(derivative).all()


def test_collocation_three_nodes_recover_independent_polynomial():
    x = torch.tensor((0.5, 0.75, 1.0), dtype=torch.float64)
    base = torch.tensor((0.13, -0.07, 0.23), dtype=torch.float64)
    coefficients = torch.tensor((0.037, -0.021, 0.013), dtype=torch.float64)
    values = base + sum(
        coefficient * x**power
        for coefficient, power in zip(coefficients, (7 / 3, 8 / 3, 3))
    )
    recovered = collocation_coefficients(base, values, 17.0)
    torch.testing.assert_close(recovered, coefficients, rtol=2e-12, atol=2e-13)


def test_collocation_explicit_two_nodes_reduces_degree_and_zero_pads():
    x = torch.tensor((0.5, 0.75, 1.0), dtype=torch.float64)
    base = torch.tensor((0.17, -0.11, 0.29), dtype=torch.float64)
    coefficients = torch.tensor((0.041, -0.026), dtype=torch.float64)
    values = base + coefficients[0] * x ** (7 / 3) + coefficients[1] * x ** (8 / 3)
    recovered = collocation_coefficients(
        base, values, 17.0, retained_nodes=(0, 2)
    )
    expected = torch.cat((coefficients, torch.zeros(1, dtype=torch.float64)))
    torch.testing.assert_close(recovered, expected, rtol=2e-12, atol=2e-13)


def test_collocation_one_node_and_zero_nodes_return_three_coefficients():
    x = torch.tensor((0.5, 0.75, 1.0), dtype=torch.float64)
    base = torch.tensor((0.13, -0.07, 0.23), dtype=torch.float64)
    values = base + 0.052 * x ** (7 / 3)
    one_node = collocation_coefficients(base, values, 17.0, retained_nodes=(1,))
    torch.testing.assert_close(
        one_node,
        torch.tensor((0.052, 0.0, 0.0), dtype=torch.float64),
        rtol=2e-12,
        atol=2e-13,
    )

    zero_nodes = collocation_coefficients(base, values, 17.0, retained_nodes=())
    torch.testing.assert_close(zero_nodes, torch.zeros(3, dtype=torch.float64))


def test_collocation_batch_and_explicit_node_selection():
    x = torch.tensor((0.5, 0.75, 1.0), dtype=torch.float64)
    base = torch.tensor(
        ((0.13, -0.07, 0.23), (0.31, 0.04, -0.12)), dtype=torch.float64
    )
    coefficients = torch.tensor(
        ((0.037, -0.021, 0.013), (-0.014, 0.028, 0.019)), dtype=torch.float64
    )
    values = base + coefficients[:, 0, None] * x ** (7 / 3)
    values = values + coefficients[:, 1, None] * x ** (8 / 3)
    values = values + coefficients[:, 2, None] * x**3
    recovered = collocation_coefficients(base, values, 17.0)
    torch.testing.assert_close(recovered, coefficients, rtol=2e-12, atol=2e-13)

    reduced_values = base + coefficients[:, 0, None] * x ** (7 / 3)
    reduced_values = reduced_values + coefficients[:, 1, None] * x ** (8 / 3)
    reduced = collocation_coefficients(
        base, reduced_values, 17.0, retained_nodes=(2, 0)
    )
    expected_reduced = torch.cat(
        (coefficients[:, :2], torch.zeros((2, 1), dtype=torch.float64)), dim=-1
    )
    torch.testing.assert_close(reduced, expected_reduced, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"active_terms": 4},
        {"active_terms": 2, "retained_nodes": (0,)},
        {"active_terms": 1, "retained_nodes": (0, 0)},
        {"retained_nodes": (3,)},
        {"retained_nodes": (-1,)},
        {"retained_nodes": (1.5,)},
    ),
)
def test_collocation_node_index_validation(kwargs):
    base = torch.zeros(3, dtype=torch.float64)
    values = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        collocation_coefficients(base, values, 17.0, **kwargs)
