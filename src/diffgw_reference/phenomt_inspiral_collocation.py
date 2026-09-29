"""Scalar check of the IMRPhenomTHM (2,2) frequency collocation solve.

Estellés et al., arXiv:2012.11923, Eqs. (8)--(11).  The six TaylorT3
coefficients and five fitted frequencies are inputs: this checks their
assembly into the pseudo-PN system, not their calibration or PN provenance.
The paper's Newtonian frequency is orbital; the (2,2) frequency is twice it.
"""

import mpmath as mp


THETA_NODES = ("0.33", "0.45", "0.55", "0.65", "0.75", "0.82")


def frequency(theta, pn_coefficients, pseudo_pn_coefficients=(), *, dps=80):
    """Evaluate the (2,2) TaylorT3 frequency and optional Eq. (9) terms."""
    if len(pn_coefficients) != 6 or len(pseudo_pn_coefficients) not in (0, 6):
        raise ValueError("expected six TaylorT3 and zero or six pseudo-PN coefficients")
    with mp.workdps(dps):
        x = mp.mpf(str(theta))
        c = [mp.mpf(str(value)) for value in pn_coefficients]
        series = 1 + sum(c[j] * x ** (j + 2) for j in range(4))
        series += c[4] * x**6 + mp.mpf(107) / 280 * mp.log(x) * x**6 + c[5] * x**7
        series += sum(mp.mpf(str(value)) * x ** (8 + j)
                      for j, value in enumerate(pseudo_pn_coefficients))
        return x**3 / 4 * series


def solve(eta, t0, fitted_frequencies, pn_coefficients, *, dps=80):
    """Solve THM Eq. (10), including the first node's TaylorT3 time shift."""
    if len(fitted_frequencies) != 5 or len(pn_coefficients) != 6:
        raise ValueError("expected five fitted frequencies and six TaylorT3 coefficients")
    with mp.workdps(dps):
        nodes = [mp.mpf(value) for value in THETA_NODES]
        eta = mp.mpf(str(eta))
        t0 = mp.mpf(str(t0))
        t_early = -5 / (eta * nodes[0] ** 8)
        theta0 = (eta * (t0 - t_early) / 5) ** (-mp.mpf(1) / 8)
        targets = [frequency(theta0, pn_coefficients, dps=dps)]
        targets += [mp.mpf(str(value)) for value in fitted_frequencies]
        matrix = mp.matrix([[x ** (8 + j) for j in range(6)] for x in nodes])
        rhs = mp.matrix([4 * (target - frequency(x, pn_coefficients, dps=dps)) / x**3
                         for x, target in zip(nodes, targets)])
        return tuple(mp.lu_solve(matrix, rhs)), tuple(targets)
