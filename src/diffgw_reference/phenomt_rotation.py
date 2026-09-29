"""Independent scalar rotation for the IMRPhenomTP/TPHM mode sum.

Estellés et al., arXiv:2105.05872, Eqs. (4), (8), and (9) rotate the
coprecessing modes with Wigner D matrices and project them on spin-weighted
harmonics.  Here the small-d matrix is obtained by exponentiating the angular
momentum generator, rather than using the factorial sum in either backend.
The supplied modes and angles are inputs: this module does not certify their
carrier or precession evolution.  No LAL implementation or coefficients are
used.
"""

import mpmath as mp


def small_d_matrix(ell: int, beta: float, *, dps: int = 60) -> mp.matrix:
    """Return ``d^ell_{m,m'}(beta)`` in ascending magnetic-index order."""
    if ell < 2 or dps < 30:
        raise ValueError("ell must be >= 2 and dps must be >= 30")
    with mp.workdps(dps):
        size = 2 * ell + 1
        raising = mp.matrix(size)
        for m in range(-ell, ell):
            raising[m + 1 + ell, m + ell] = mp.sqrt((ell - m) * (ell + m + 1))
        jy = (raising - raising.transpose()) / (2j)
        return mp.expm(-1j * mp.mpf(str(beta)) * jy)


def rotation_matrix(
    ell: int, alpha: float, beta: float, gamma: float, *, dps: int = 60
) -> mp.matrix:
    """Return the active coprecessing-to-J rotation in Eq. (4)."""
    with mp.workdps(dps):
        d = small_d_matrix(ell, beta, dps=dps)
        left = mp.matrix([mp.exp(-1j * m * mp.mpf(str(alpha))) for m in range(-ell, ell + 1)])
        right = mp.matrix([mp.exp(-1j * m * mp.mpf(str(gamma))) for m in range(-ell, ell + 1)])
        return mp.matrix([[left[m + ell] * d[m + ell, n + ell] * right[n + ell]
                           for n in range(-ell, ell + 1)]
                          for m in range(-ell, ell + 1)])


def project_modes(
    modes: dict[tuple[int, int], complex],
    alpha: float,
    beta: float,
    gamma: float,
    theta_jn: float,
    zeta: float,
    *,
    dps: int = 60,
) -> complex:
    """Return ``h_plus - i*h_cross`` for supplied coprecessing modes.

    When only a positive-m mode is supplied, use the nonprecessing-mode
    symmetry ``h_l,-m = (-1)^l conjugate(h_l,m)``.  An explicitly supplied
    negative-m mode takes precedence.
    """
    if not modes:
        raise ValueError("at least one mode is required")
    full = dict(modes)
    for (ell, m), value in modes.items():
        if m > 0:
            full.setdefault((ell, -m), (-1) ** ell * complex(value).conjugate())
    with mp.workdps(dps):
        total = mp.mpc(0)
        for ell in sorted({ell for ell, _ in full}):
            rotated = rotation_matrix(ell, alpha, beta, gamma, dps=dps)
            harmonics = small_d_matrix(ell, theta_jn, dps=dps)
            normalization = mp.sqrt((2 * ell + 1) / (4 * mp.pi))
            for m in range(-ell, ell + 1):
                j_mode = sum(
                    rotated[m + ell, n + ell] * mp.mpc(full.get((ell, n), 0))
                    for n in range(-ell, ell + 1)
                )
                total += normalization * harmonics[m + ell, 2 + ell] * j_mode
        total *= mp.exp(2j * mp.mpf(str(zeta)))
        return complex(total)


__all__ = ["small_d_matrix", "rotation_matrix", "project_modes"]
