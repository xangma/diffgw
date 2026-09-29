"""Independent supplied-input IMRPhenomT merger equations.

Estellés et al., arXiv:2012.11923v1, Eqs. (28)–(32). The scalar
matching systems use the paper's basis and import no carrier or fit code.
The dominant-mode frequency collocation time is a carrier input: the paper's
generic -25 M is not substituted for the native (2,2) theta=0.95 choice.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "phenomt-merger/v1"
FREQUENCY_COMPONENT_ID = "T-MERGER-22-FREQ-v1"
AMPLITUDE_COMPONENT_ID = "T-MERGER-22-AMP-v1"
SOURCE = "Estelles et al., arXiv:2012.11923v1, Eqs. (28)-(32)"


def symbolic_ansatz():
    """Return the paper's two merger expressions as SymPy objects.

    SymPy is imported only for symbolic audits; numerical reference calls
    require only mpmath and never import production carrier code.
    """
    import sympy as sp

    t, alpha, peak_time, omega_rd = sp.symbols(
        "t alpha peak_time omega_rd", real=True
    )
    a = sp.symbols("a0:5", real=True)
    b = sp.symbols("b0:4", real=True)
    x = sp.asinh(alpha * t)
    tau = t - peak_time
    omega = omega_rd * (1 - sum(a[k] * x**k for k in range(5)))
    amplitude = (
        b[0] + b[1] * tau**2
        + b[2] * sp.sech(2 * alpha * tau) ** sp.Rational(1, 7)
        + b[3] * sp.sech(alpha * tau)
    )
    return (t, alpha, peak_time, omega_rd), a, b, omega, amplitude


def _mp(*values):
    return tuple(mp.mpf(str(value)) for value in values)


def _frequency_basis(t, alpha):
    x = mp.asinh(alpha * t)
    return (mp.mpf(1), x, x * x, x**3, x**4)


def _frequency_slope_basis(t, alpha):
    x = mp.asinh(alpha * t)
    dx = alpha / mp.sqrt(1 + (alpha * t) ** 2)
    return (mp.mpf(0), dx, 2 * x * dx, 3 * x * x * dx, 4 * x**3 * dx)


def frequency_coefficients(t_cut, t_mid, omega_rd, alpha, omega_cut,
                           slope_cut, omega_mid, omega_peak, slope_peak,
                           *, dps=80):
    """Return Eq. (29) coefficients a0..a4 from Eq. (31) conditions."""
    with mp.workdps(dps):
        cut, mid, wr, rate, wc, dc, wm, wp, dp = _mp(
            t_cut, t_mid, omega_rd, alpha, omega_cut,
            slope_cut, omega_mid, omega_peak, slope_peak
        )
        if not all(map(mp.isfinite, (cut, mid, wr, rate, wc, dc, wm, wp, dp))):
            raise ValueError("merger frequency inputs must be finite")
        if not cut < mid < 0 or wr <= 0 or rate <= 0:
            raise ValueError("requires t_cut<t_mid<0, omega_rd>0, alpha>0")
        matrix = mp.matrix((
            _frequency_basis(cut, rate),
            _frequency_basis(mid, rate),
            _frequency_basis(0, rate),
            _frequency_slope_basis(cut, rate),
            _frequency_slope_basis(0, rate),
        ))
        rhs = mp.matrix((1 - wc / wr, 1 - wm / wr, 1 - wp / wr,
                         -dc / wr, -dp / wr))
        solution = mp.lu_solve(matrix, rhs)
        return tuple(solution)


def frequency(t, omega_rd, alpha, coefficients, *, dps=80):
    """Eq. (28) angular frequency within the supplied merger interval."""
    with mp.workdps(dps):
        time, wr, rate = _mp(t, omega_rd, alpha)
        return wr * (1 - sum(c * b for c, b in zip(coefficients, _frequency_basis(time, rate))))


def frequency_slope(t, omega_rd, alpha, coefficients, *, dps=80):
    with mp.workdps(dps):
        time, wr, rate = _mp(t, omega_rd, alpha)
        return -wr * sum(c * b for c, b in zip(coefficients, _frequency_slope_basis(time, rate)))


def phase_increment(t, origin, omega_rd, alpha, coefficients, *, dps=80):
    """Independent high-precision integral of the Eq. (28) frequency."""
    with mp.workdps(dps):
        time, start = _mp(t, origin)
        return mp.quad(lambda u: frequency(u, omega_rd, alpha, coefficients, dps=dps),
                       [start, time])


def _amplitude_basis(t, alpha, peak_time):
    tau = t - peak_time
    return (mp.mpf(1), tau * tau,
            mp.cosh(2 * alpha * tau) ** (-mp.mpf(1) / 7),
            1 / mp.cosh(alpha * tau))


def _amplitude_slope_basis(t, alpha, peak_time):
    tau = t - peak_time
    two = 2 * alpha * tau
    one = alpha * tau
    return (mp.mpf(0), 2 * tau,
            -2 * alpha * mp.tanh(two) * mp.cosh(two) ** (-mp.mpf(1) / 7) / 7,
            -alpha * mp.tanh(one) / mp.cosh(one))


def amplitude_coefficients(t_cut, t_mid, peak_time, alpha, amplitude_cut,
                           slope_cut, amplitude_mid, amplitude_peak, *, dps=80):
    """Return Eq. (30) coefficients b0..b3 from Eq. (32) conditions."""
    with mp.workdps(dps):
        cut, mid, peak_t, rate, hc, dc, hm, hp = _mp(
            t_cut, t_mid, peak_time, alpha, amplitude_cut,
            slope_cut, amplitude_mid, amplitude_peak
        )
        if not all(map(mp.isfinite, (cut, mid, peak_t, rate, hc, dc, hm, hp))):
            raise ValueError("merger amplitude inputs must be finite")
        if not cut < mid < peak_t or rate <= 0:
            raise ValueError("requires t_cut<t_mid<t_peak and alpha>0")
        matrix = mp.matrix((
            _amplitude_basis(cut, rate, peak_t),
            _amplitude_basis(mid, rate, peak_t),
            _amplitude_basis(peak_t, rate, peak_t),
            _amplitude_slope_basis(cut, rate, peak_t),
        ))
        solution = mp.lu_solve(matrix, mp.matrix((hc, hm, hp, dc)))
        return tuple(solution)


def amplitude(t, alpha, peak_time, coefficients, *, dps=80):
    with mp.workdps(dps):
        time, rate, peak_t = _mp(t, alpha, peak_time)
        return sum(c * b for c, b in zip(coefficients, _amplitude_basis(time, rate, peak_t)))


def amplitude_slope(t, alpha, peak_time, coefficients, *, dps=80):
    with mp.workdps(dps):
        time, rate, peak_t = _mp(t, alpha, peak_time)
        return sum(c * b for c, b in zip(coefficients, _amplitude_slope_basis(time, rate, peak_t)))
