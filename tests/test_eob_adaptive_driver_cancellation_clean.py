from fractions import Fraction

import torch

import torchwave.approximants.eob.adaptive_driver_clean as adaptive_driver


DTYPE = torch.float64


def _public_probe_field(time, state):
    """Small independent field used to expose embedded-error cancellation."""
    return torch.stack((
        state[..., 0],
        0.02 * torch.sin(time) + 0.03 * torch.ones_like(time),
    ), dim=-1)


def test_rkf45_embedded_error_is_invariant_to_large_state_translation():
    offsets = torch.tensor([0.0, 132.0, 1.0e6, 1.0e10], dtype=DTYPE)
    time = torch.full_like(offsets, 0.3)
    step = torch.full_like(offsets, 0.1)
    state = torch.stack((torch.ones_like(offsets), offsets), dim=-1)
    current = _public_probe_field(time, state)

    _, _, error, evaluations = adaptive_driver._rkf45_trial(
        _public_probe_field,
        time,
        state,
        step,
        current,
        2,
    )

    # The second component of the field is independent of the translated
    # state component.  Direct stage-weighted error arithmetic therefore
    # gives the same local error at every offset, including 1e10.
    torch.testing.assert_close(
        error[:, 1],
        error[0, 1].expand_as(error[:, 1]),
        rtol=0.0,
        atol=2.0e-18,
    )
    assert abs(float(error[0, 1])) > 1.0e-13
    assert evaluations == 5


def test_rkf45_embedded_weights_match_independent_fraction_subtraction():
    b4 = (
        Fraction(25, 216), Fraction(0), Fraction(1408, 2565),
        Fraction(2197, 4104), Fraction(-1, 5), Fraction(0),
    )
    b5 = (
        Fraction(16, 135), Fraction(0), Fraction(6656, 12825),
        Fraction(28561, 56430), Fraction(-9, 50), Fraction(2, 55),
    )
    expected = tuple(float(high - low) for high, low in zip(b5, b4))
    assert adaptive_driver._RKF45_B5_MINUS_B4 == expected

