"""Checks for the independently generated numerical D QNM representation."""

import pytest
import torch

from torchwave.approximants.kerr_qnm_d_sampling_clean import (
    COMPATIBILITY_SPIN_MAX,
    COMPATIBILITY_SPIN_MIN,
    GENERATED_SPIN_LIMIT,
    KNOT_SPINS,
    NATURAL_CUBIC_COEFFICIENTS,
    N_KNOTS,
    ROUNDED_CYCLIC_ORDINATES,
    SPIN_STEP,
    _round_significant,
    _rounding_step,
    _evaluate_cubic,
    evaluate_d_qnm_220,
    evaluate_d_qnm_220_generated,
    independent_cyclic_qnm_220,
)


def test_grid_has_uniform_interior_and_explicit_generated_guards():
    assert N_KNOTS == 1001
    assert KNOT_SPINS[0] == pytest.approx(-0.999)
    assert KNOT_SPINS[-1] == pytest.approx(0.999)
    assert KNOT_SPINS[1] == pytest.approx(-0.998)
    assert KNOT_SPINS[-2] == pytest.approx(0.998)
    assert all(
        (right - left) == pytest.approx(SPIN_STEP)
        for left, right in zip(KNOT_SPINS[1:-2], KNOT_SPINS[2:-1])
    )
    assert COMPATIBILITY_SPIN_MIN == pytest.approx(-0.994)
    assert COMPATIBILITY_SPIN_MAX == pytest.approx(0.994)


def test_ordinates_are_six_significant_digit_roundings_of_generated_cyclic_values():
    spins = torch.tensor(KNOT_SPINS, dtype=torch.float64)
    real, damping = independent_cyclic_qnm_220(spins)
    raw = torch.stack((real, damping), dim=-1)
    rounded = torch.tensor(ROUNDED_CYCLIC_ORDINATES, dtype=torch.float64)

    for raw_value, rounded_value in zip(raw.reshape(-1), rounded.reshape(-1)):
        raw_float = float(raw_value)
        expected = _round_significant(raw_float)
        assert float(rounded_value) == expected
        # The interval check catches an accidental truncation or a rounding
        # operation performed before converting angular roots to cyclic units.
        assert abs(raw_float - float(rounded_value)) <= _rounding_step(raw_float) + 1e-15


def test_rounded_knots_are_reproduced_by_the_cubic():
    spins = torch.tensor(KNOT_SPINS, dtype=torch.float64)
    real, damping = _evaluate_cubic(spins)
    values = torch.stack((real, damping), dim=-1)
    expected = torch.tensor(ROUNDED_CYCLIC_ORDINATES, dtype=torch.float64)
    torch.testing.assert_close(values, expected, rtol=0.0, atol=2e-15)


def test_natural_cubic_is_c1_and_has_zero_endpoint_curvature():
    coefficients = torch.tensor(NATURAL_CUBIC_COEFFICIENTS, dtype=torch.float64)
    knots = torch.tensor(KNOT_SPINS, dtype=torch.float64)
    widths = knots[1:] - knots[:-1]

    # The value and first and second derivatives agree at every internal knot.
    for index in range(1, N_KNOTS - 1):
        left = coefficients[index - 1]
        right = coefficients[index]
        width = widths[index - 1]
        left_first = left[1] + 2.0 * left[2] * width + 3.0 * left[3] * width.square()
        right_first = right[1]
        left_second = 2.0 * left[2] + 6.0 * left[3] * width
        right_second = 2.0 * right[2]
        torch.testing.assert_close(left_first, right_first, rtol=0.0, atol=3e-12)
        torch.testing.assert_close(left_second, right_second, rtol=0.0, atol=3e-10)

    first_second = 2.0 * coefficients[0, 2]
    last_width = widths[-1]
    last = coefficients[-1]
    last_second = 2.0 * last[2] + 6.0 * last[3] * last_width
    torch.testing.assert_close(
        first_second, torch.zeros(2, dtype=torch.float64), rtol=0.0, atol=1e-15
    )
    torch.testing.assert_close(
        last_second, torch.zeros(2, dtype=torch.float64), rtol=0.0, atol=3e-12
    )


def test_public_evaluator_preserves_shape_dtype_and_autograd():
    spin = torch.tensor([[-0.7312, -0.1234], [0.1234, 0.7312]], dtype=torch.float64)
    spin.requires_grad_()
    real, damping = evaluate_d_qnm_220(spin)
    assert real.shape == spin.shape
    assert damping.shape == spin.shape
    assert real.dtype == spin.dtype
    assert damping.dtype == spin.dtype
    gradient = torch.autograd.grad((real + damping).sum(), spin)[0]
    assert torch.isfinite(gradient).all()

    spin32 = torch.tensor([-0.7, 0.0, 0.7], dtype=torch.float32)
    real32, damping32 = evaluate_d_qnm_220(spin32)
    assert real32.dtype == torch.float32
    assert damping32.dtype == torch.float32


def test_inference_first_does_not_poison_later_autograd():
    probe = torch.tensor([-0.6, 0.2], dtype=torch.float64)
    with torch.inference_mode():
        inferred = evaluate_d_qnm_220(probe)
    differentiable = probe.detach().requires_grad_()
    real, damping = evaluate_d_qnm_220(differentiable)
    gradient = torch.autograd.grad((real + damping).sum(), differentiable)[0]
    assert all(torch.isfinite(value).all() for value in inferred)
    assert torch.isfinite(real).all() and torch.isfinite(damping).all()
    assert torch.isfinite(gradient).all()


def test_gradcheck_away_from_knot_and_fallback_boundary():
    interior = torch.tensor([-0.7312, 0.1234], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: torch.stack(evaluate_d_qnm_220(value)),
        (interior,),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-6,
    )

    generated = torch.tensor([0.9963, -0.9971], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: torch.stack(evaluate_d_qnm_220(value)),
        (generated,),
        eps=1e-6,
        atol=2e-7,
        rtol=2e-6,
    )
    table = evaluate_d_qnm_220(torch.tensor([0.9939, -0.9939], dtype=torch.float64))
    generated_direct = evaluate_d_qnm_220_generated(
        torch.tensor([0.9939, -0.9939], dtype=torch.float64)
    )
    assert any(torch.max(torch.abs(a - b)) > 1e-12 for a, b in zip(table, generated_direct))


def test_generated_fallback_is_used_outside_compatibility_interior():
    query = torch.tensor([-0.998, -0.995, 0.0, 0.995, 0.998], dtype=torch.float64)
    rounded = evaluate_d_qnm_220(query)
    generated = evaluate_d_qnm_220_generated(query)
    for channel in range(2):
        torch.testing.assert_close(
            rounded[channel][[0, 1, 3, 4]],
            generated[channel][[0, 1, 3, 4]],
            rtol=0.0,
            atol=0.0,
        )
    assert torch.max(torch.abs(rounded[0][2] - generated[0][2])) > 1e-12


def test_unsupported_extremal_and_invalid_inputs_are_explicit():
    with pytest.raises(TypeError, match="floating dtype"):
        evaluate_d_qnm_220(torch.tensor(0))
    with pytest.raises(ValueError, match="extremal"):
        evaluate_d_qnm_220(torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="extremal"):
        evaluate_d_qnm_220(torch.tensor(-1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="supports"):
        evaluate_d_qnm_220(torch.tensor(GENERATED_SPIN_LIMIT + 1e-6))
    with pytest.raises(ValueError, match="supports"):
        evaluate_d_qnm_220(torch.tensor(float("nan"), dtype=torch.float64))
