"""Small clean NSBH amplitude helpers from the public frequency convention."""

import torch


def positive_tidal_frequency(denominator: torch.Tensor) -> torch.Tensor:
    """Return the positive-frequency magnitude of Thompson et al. Eq. (8).

    Eq. (8) writes the signed algebraic denominator.  Callers provide the
    complete denominator, including its pi factor.  The one-sided positive
    frequency amplitude convention used by the public binary output selects
    its magnitude when that denominator is negative (notably at C=1/2 and
    negative spin).
    """
    return torch.abs(torch.reciprocal(denominator))


def delta2_prime_plus(f_tide: torch.Tensor, f_rd_tilde: torch.Tensor) -> torch.Tensor:
    r"""Return the NSBH delta2 numerical-constant smooth plus branch.

    Thompson (2020) Eq. (29) prints the minus window (\omega^-):
      (A/2) * \omega^-_{x_3, d_3}(x) = 0.81248 * 0.5 * (1 - \tanh(4*(x - x_3)/d_3))
    which decays to zero in the non-disruptive BBH limit (f_tide >> f_rd_tilde).

    The project_correction variant implemented here uses the smooth plus window
    (+0.81248 \omega^+):
      0.81248 * 0.5 * (1 + \tanh(4*(x - 0.0188092) / 0.338737))
    which preserves the BBH ringdown width constant \delta_2 = 0.81248 as
    f_tide -> +\infty.
    """
    x = (f_tide - f_rd_tilde) / f_rd_tilde
    return 0.81248 * 0.5 * (1.0 + torch.tanh(4.0 * (x - 0.0188092) / 0.338737))


def delta2_prime_small_lambda(delta2: torch.Tensor, lambda2: torch.Tensor) -> torch.Tensor:
    """Thompson Eq. (29), closed 0<=Lambda<=1 branch with supplied delta2."""
    return delta2 - 2.0 * (delta2 - 0.81248) * lambda2 + (delta2 - 0.81248) * (lambda2 ** 2.0)


def gamma1_prime_small_lambda(lambda2: torch.Tensor) -> torch.Tensor:
    r"""Continuous quadratic for the Eq. 26 small-tidal branch (project_correction variant).

    Thompson et al. (2020) Eq. (26) prints:
      \gamma'_1 / \gamma_1 = 1 - 0.5*\Lambda - 0.25*\Lambda^2   for \Lambda <= 1
      \gamma'_1 / \gamma_1 = 1.25                               for \Lambda > 1
    which evaluates to 0.25 at \Lambda = 1, creating an unphysical jump discontinuity
    of \Delta = +1.0 at \Lambda = 1.

    The smooth polynomial continuation variant (labeled 'project_correction' or
    'compatibility') implemented here is:
      g(\Lambda) = 1 + 0.5*\Lambda - 0.25*\Lambda^2
    It is the unique quadratic satisfying:
      - g(0) = 1.0 (smooth connection to BBH baseline)
      - g(1) = 1.25 (matching the \Lambda > 1 constant)
      - g'(1) = 0.5 - 0.5(1) = 0.0 (C^1 continuous match to \Lambda > 1).
    """
    return 1.0 + 0.5 * lambda2 - 0.25 * (lambda2 ** 2.0)


def outer_pn_center(base_center: torch.Tensor, epsilon_ins: torch.Tensor) -> torch.Tensor:
    """Apply epsilon_ins to the complete mixed PN center.

    This exact composition is a black-box compatibility result inspired by
    the general Eq. 9 outer epsilon factor.  Eq. 28/Table 1 literally give
    an already mixed center, so this helper records the compatibility
    convention rather than claiming that the extra multiplication is
    printed there. Callers select it only for the mildly disruptive,
    no-torus branch.
    """
    return epsilon_ins * base_center


__all__ = [
    "positive_tidal_frequency",
    "delta2_prime_plus",
    "delta2_prime_small_lambda",
    "gamma1_prime_small_lambda",
    "outer_pn_center",
]
