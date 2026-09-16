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
    """Return the NSBH delta2 numerical-constant smooth plus branch.

    The printed 2020 Eq. 29 uses the opposite ``omega_minus`` sign.  This
    sign is selected here from the advertised large-tidal-frequency BBH
    continuity limit and is corroborated by the public
    ``SimIMRPhenomNSBH_delta2_prime`` callable.  No oracle fit is used.
    """
    x = (f_tide - f_rd_tilde) / f_rd_tilde
    return 0.81248 * 0.5 * (1.0 + torch.tanh(4.0 * (x - 0.0188092) / 0.338737))


def gamma1_prime_small_lambda(lambda2: torch.Tensor) -> torch.Tensor:
    """Continuous quadratic for the Eq. 26 small-tidal branch.

    It is the unique quadratic satisfying g(0)=1, g(1)=1.25 and g'(1)=0;
    this exposes the sign typo in the printed small-Lambda expression while
    preserving the stated boundary continuity.
    """
    return 1.0 + 0.5 * lambda2 - 0.25 * lambda2**2


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


__all__ = ["positive_tidal_frequency", "delta2_prime_plus", "gamma1_prime_small_lambda", "outer_pn_center"]
