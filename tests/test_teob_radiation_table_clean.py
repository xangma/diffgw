"""Equation-level checks for the clean TEOB residual-amplitude completion."""

import math

import torch

from torchwave.approximants.eob.factorized_modes import _rho_table
from torchwave.approximants.eob.teob_radiation_clean import _complete_rho_table
from torchwave.constants import EULER_GAMMA


def _eulerlog(x: torch.Tensor, m: int) -> torch.Tensor:
    return EULER_GAMMA + math.log(2 * m) + 0.5 * torch.log(x)


def test_finite_mass_3pn_octupoles_are_additive_to_the_clean_base():
    """Check the nu-dependent x**3 terms in arXiv:1409.3546v2 Eq. (64)."""
    x = torch.tensor(0.037, dtype=torch.float64)
    nu = torch.tensor(0.23, dtype=torch.float64)
    zero = torch.zeros_like(nu)
    base = _rho_table(x, nu)
    complete = _complete_rho_table(x, nu)
    complete_zero = _complete_rho_table(x, zero)

    expected = {
        (3, 3): ((-129509 / 25740 + 41 * math.pi**2 / 192) * nu
                 - 274621 / 154440 * nu**2 + 12011 / 46332 * nu**3) * x**3,
        (3, 1): ((-9688441 / 2084940 + 41 * math.pi**2 / 192) * nu
                 + 174535 / 75816 * nu**2 - 727247 / 1250964 * nu**3) * x**3,
    }
    for mode, term in expected.items():
        # The extra test-mass x**5 term is independent of nu, so differencing
        # the assembled and nu=0 tables isolates the finite-mass equation.
        assert torch.allclose(
            (complete[mode] - base[mode])
            - (complete_zero[mode] - _rho_table(x, zero)[mode]),
            term,
            rtol=2e-14,
            atol=2e-16,
        )

    # At nu=0 the Eq. (64) additions vanish exactly; only the already
    # published test-mass x**5 completions remain on these two rows.
    assert torch.allclose(
        complete_zero[(3, 3)] - _rho_table(x, zero)[(3, 3)],
        (-903823148417327 / 30566888352000
         + 87347 / 13860 * _eulerlog(x, 3)) * x**5,
        rtol=2e-14,
        atol=2e-16,
    )
    assert torch.allclose(
        complete_zero[(3, 1)] - _rho_table(x, zero)[(3, 1)],
        (430750057673539 / 297110154781440
         - 1313 / 224532 * _eulerlog(x, 1)) * x**5,
        rtol=2e-14,
        atol=2e-16,
    )


def test_published_test_mass_3pn_rows_and_l8_2pn_rows():
    """Check Appendix D6, D8, D9 and D27--D33 without order double-counts."""
    x = torch.tensor(0.041, dtype=torch.float64)
    nu = torch.tensor(0.19, dtype=torch.float64)
    base = _rho_table(x, nu)
    complete = _complete_rho_table(x, nu)

    expected = {
        (4, 3): (
            (-1571 / 770 * _eulerlog(x, 3) + 1664224207351 / 195343948800) * x**3
            + (-2465107182496333 / 460490801971200
               + 174381 / 67760 * _eulerlog(x, 3)) * x**4
        ),
        (4, 1): (
            (-1571 / 6930 * _eulerlog(x, 1) + 1227423222031 / 1758095539200) * x**3
            + (-29584392078751453 / 37299754959667200
               + 67553 / 261360 * _eulerlog(x, 1)) * x**4
        ),
        (5, 5): (
            (-1546 / 429 * _eulerlog(x, 5) + 190606537999247 / 11957879934000) * x**3
            + (-1213641959949291437 / 118143853747920000
               + 376451 / 83655 * _eulerlog(x, 5)) * x**4
        ),
    }
    for mode, term in expected.items():
        assert torch.allclose(
            complete[mode] - base[mode], term, rtol=2e-14, atol=2e-16
        )

    expected_l8 = {
        (8, 8): -1.5337092502821381,
        (8, 7): -1.175404252991305,
        (8, 6): -0.9061610303170207,
        (8, 5): -0.7220789990670207,
        (8, 4): -0.47652059150068155,
        (8, 3): -0.4196774909106648,
        (8, 2): -0.2261796441029474,
        (8, 1): -0.26842133517043704,
    }
    for mode, coefficient in expected_l8.items():
        assert torch.allclose(
            complete[mode] - base[mode], coefficient * x**2,
            rtol=2e-14,
            atol=2e-16,
        )


def test_table_completion_keeps_gradients_through_new_terms():
    x = torch.tensor(0.031, dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    values = _complete_rho_table(x, nu)
    loss = values[(3, 3)] + values[(4, 3)] + values[(5, 5)] + values[(8, 8)]
    dx, dnu = torch.autograd.grad(loss, (x, nu))
    assert torch.isfinite(dx) and torch.isfinite(dnu)
