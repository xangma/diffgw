"""Nonspinning, quasi-circular point-particle TaylorF2 through 3.5PN.

The canonical expressions use exact SymPy rationals. Scalar evaluation walks
those expressions with a private mpmath context; it never imports production
waveforms, constants, coefficient tables, or another waveform implementation.

Sources are pinned below. Buonanno et al. (2009), Eq. (3.18), prints
``log(v/v_lso)`` at 2.5PN, with ``v_lso = 1/sqrt(6)``. ``phase_expr`` instead
uses ``log(v)``: a constant phase gauge, not a different PN coefficient.
Precisely, ``I_paper = I_logv - 9*phi5*log(v_lso)/(128*eta)``. The gravitational
wave coalescence phase changes by the opposite compensating convention:
``Phi_c_logv = Phi_c_paper + 9*phi5*log(v_lso)/(128*eta)``.
``paper_phase_expr`` retains the source's intrinsic phase gauge.

The waveform adapter uses seconds for total mass M=G*(m1+m2)/c**3 and
distance D=d/c, hertz for f, radians for inclination and orbital phic, and
seconds for tc. Its Fourier transform is integral h(t)*exp(-2*pi*i*f*t) dt.
Arun et al. (2008), Eqs. (6.13) and (D1), at n=0,k=2 give a negative plus
amplitude and negative imaginary cross amplitude in the opposite Fourier
convention. Conjugating and absorbing the overall minus into the exponential
gives a source-basis phase I_logv + 2*pi*f*tc - 2*phic + 3*pi/4. The
project adapter chooses an additional constant phase offset in coalescence mode:
Psi = I_logv + 2*pi*f*tc - 2*phic - pi/4,
h_plus = (1+cos(iota)**2)/2*A*exp(-i*Psi),
h_cross = -i*cos(iota)*A*exp(-i*Psi).
This additional pi offset is a phase-origin convention (equivalently, an orbital
phase shift of pi/2), not the source's polarization sign at the same phase.
Finite-reference mode instead uses Psi = I_logv(f) - I_logv(f_ref)
+ 2*pi*f*tc - 2*phic + 3*pi/4. Its subtraction and constant offset are explicit
project conventions, not an orbital phase at f_ref attributed to those papers.

Only positive frequencies are evaluated. There are no spins, tides, eccentricity,
higher amplitude terms, merger model, cutoffs, tapering, or detector response.
The analytic domain is v>0, 0<eta<=1/4, M>0, D>0, f>0, 0<=iota<=pi.
The PN approximation is intended for slow inspiral; the analytic domain is not
a claim of physical accuracy throughout that domain. Derivatives hold all other
arguments fixed and apply only to these smooth, unwindowed expressions.
"""

from __future__ import annotations

from collections.abc import Mapping

import mpmath
import sympy as sp

SPEC_VERSION = "taylorf2-nonspinning-3.5pn/v1"
REFERENCE_VERSION = SPEC_VERSION

SOURCES = {
    "buonanno2009": {
        "arxiv": "0907.0700v1",
        "url": "https://arxiv.org/pdf/0907.0700v1",
        "doi": "10.1103/PhysRevD.80.084043",
        "equations": ("3.14", "3.17", "3.18"),
        "use": "Nonspinning 3.5PN intrinsic phase; positive Fourier convention.",
    },
    "arun2008": {
        "arxiv": "0810.5336v1",
        "url": "https://arxiv.org/pdf/0810.5336v1",
        "doi": "10.1103/PhysRevD.79.104023",
        "equations": ("6.8", "6.9", "6.13", "D1"),
        "use": "Newtonian second-harmonic amplitude and polarization factors.",
    },
}

V, ETA = sp.symbols("v eta", positive=True)
F, M_SEC, D_SEC, F_REF = sp.symbols("f M_sec D_sec f_ref", positive=True)
IOTA, TC, PHIC = sp.symbols("iota tc phic", real=True)
SYMBOLS = {s.name: s for s in (V, ETA, F, M_SEC, D_SEC, F_REF, IOTA, TC, PHIC)}

COMPONENT_IDS = {
    name: f"tf2.ns35.{name}.v1"
    for name in (
        "phi0", "phi2", "phi3", "phi4", "phi5", "phi5_log", "phi6",
        "phi6_log", "phi7", "intrinsic_phase", "paper_phase", "amplitude",
        "polarizations", "coalescence_phase", "reference_phase",
    )
}
COMPONENT_SOURCES = {
    COMPONENT_IDS[name]: ("buonanno2009", "3.18")
    for name in COMPONENT_IDS
    if name.startswith("phi") or name in ("intrinsic_phase", "paper_phase")
}
COMPONENT_SOURCES.update({
    COMPONENT_IDS["amplitude"]: ("arun2008", "6.13", "D1"),
    COMPONENT_IDS["polarizations"]: ("arun2008", "6.13", "D1"),
    COMPONENT_IDS["coalescence_phase"]: ("project-convention", REFERENCE_VERSION),
    COMPONENT_IDS["reference_phase"]: ("project-convention", REFERENCE_VERSION),
})


def coefficients(eta: sp.Expr = ETA) -> dict[str, sp.Expr]:
    """Return coefficients inside 3/(128*eta*v**5), without that prefactor.

    ``phi5_log`` multiplies v**5*log(v), ``phi6_log`` v**6*log(4*v).
    ``phi6`` includes the exact Euler constant, not a rounded decimal.
    """
    eta = sp.sympify(eta)
    r = sp.Rational
    phi5 = sp.pi * (r(38645, 756) - r(65, 9) * eta)
    return {
        "phi0": sp.S.One,
        "phi2": r(20, 9) * (r(743, 336) + r(11, 4) * eta),
        "phi3": -16 * sp.pi,
        "phi4": 10 * (r(3058673, 1016064) + r(5429, 1008) * eta
                       + r(617, 144) * eta**2),
        "phi5": phi5,
        "phi5_log": 3 * phi5,
        "phi6": (r(11583231236531, 4694215680) - r(640, 3) * sp.pi**2
                 - r(6848, 21) * sp.EulerGamma
                 + (-r(15737765635, 3048192) + r(2255, 12) * sp.pi**2) * eta
                 + r(76055, 1728) * eta**2 - r(127825, 1296) * eta**3),
        "phi6_log": -r(6848, 21),
        "phi7": sp.pi * (r(77096675, 254016) + r(378515, 1512) * eta
                          - r(74045, 756) * eta**2),
    }


def phase_expr(v: sp.Expr = V, eta: sp.Expr = ETA) -> sp.Expr:
    """Intrinsic phase in the explicitly chosen log(v) coalescence gauge."""
    v, eta = sp.sympify(v), sp.sympify(eta)
    c = coefficients(eta)
    polynomial = sum(c[f"phi{k}"] * v**k for k in (0, 2, 3, 4, 5, 6, 7))
    polynomial += c["phi5_log"] * v**5 * sp.log(v)
    polynomial += c["phi6_log"] * v**6 * sp.log(4 * v)
    return sp.Rational(3, 128) * polynomial / (eta * v**5)


def paper_phase_expr(v: sp.Expr = V, eta: sp.Expr = ETA) -> sp.Expr:
    """Intrinsic part of Buonanno et al. Eq. (3.18), including its log scale."""
    eta = sp.sympify(eta)
    v_lso = 1 / sp.sqrt(6)
    shift = sp.Rational(9, 128) * coefficients(eta)["phi5"] * sp.log(v_lso) / eta
    return phase_expr(v, eta) - shift


def amplitude_expr() -> sp.Expr:
    """Positive face-on restricted Fourier amplitude A(f), in seconds.

    From Arun Eq. (6.13), n=0,k=2 and Eq. (D1): twice the prefactor
    M**2/D*sqrt(5*pi*eta/96)*v**(-7/2), with v=(pi*M*f)**(1/3).
    The source cutoff is excluded from this unwindowed reference.
    """
    return (sp.sqrt(sp.Rational(5, 24) * ETA) * sp.pi**sp.Rational(-2, 3)
            * M_SEC**sp.Rational(5, 6) / D_SEC * F**sp.Rational(-7, 6))


def waveform_expr(*, reference_frequency: bool = False) -> dict[str, sp.Expr]:
    """Return amplitude, phase, h_plus and h_cross in the adapter convention.

    Set ``reference_frequency=True`` to subtract the intrinsic phase at F_REF.
    Otherwise phic is the orbital coalescence phase in the log(v) gauge.
    In reference-frequency mode, phic is simply the adapter's phase parameter;
    at f=f_ref the total phase is 2*pi*f_ref*tc - 2*phic + 3*pi/4.
    """
    if not isinstance(reference_frequency, bool):
        raise TypeError("reference_frequency must be a boolean for waveform_expr")
    phase = phase_expr((sp.pi * M_SEC * F)**sp.Rational(1, 3), ETA)
    if reference_frequency:
        phase -= phase_expr((sp.pi * M_SEC * F_REF)**sp.Rational(1, 3), ETA)
    offset = sp.Rational(3, 4) if reference_frequency else -sp.Rational(1, 4)
    phase += 2 * sp.pi * F * TC - 2 * PHIC + offset * sp.pi
    amplitude = amplitude_expr()
    carrier = amplitude * sp.exp(-sp.I * phase)
    cosine = sp.cos(IOTA)
    return {"amplitude": amplitude, "phase": phase,
            "h_plus": (1 + cosine**2) * carrier / 2,
            "h_cross": -sp.I * cosine * carrier}


def derivative_expr(expression: sp.Expr, variable: sp.Symbol | str,
                    order: int = 1) -> sp.Expr:
    """Exact partial derivative at fixed values of the other input symbols."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError("derivative order must be a nonnegative integer")
    if isinstance(variable, str):
        variable = SYMBOLS[variable]
    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a SymPy Symbol or a canonical symbol name")
    return sp.diff(expression, variable, order)


def _walk(expression: sp.Expr, values: Mapping, ctx):
    """Evaluate the small expression language without binary64 constants."""
    if expression in values:
        return values[expression]
    if isinstance(expression, sp.Rational):
        return ctx.mpf(int(expression.p)) / int(expression.q)
    if expression == sp.pi:
        return +ctx.pi
    if expression == sp.EulerGamma:
        return +ctx.euler
    if expression == sp.I:
        return ctx.j
    if isinstance(expression, sp.Float):
        raise TypeError("use exact SymPy rationals and decimal string inputs, not Float expressions")
    args = [_walk(arg, values, ctx) for arg in expression.args]
    if expression.func is sp.Add:
        return ctx.fsum(args)
    if expression.func is sp.Mul:
        return ctx.fprod(args)
    if expression.func is sp.Pow:
        return ctx.power(*args)
    functions = {sp.log: ctx.log, sp.exp: ctx.exp, sp.sin: ctx.sin, sp.cos: ctx.cos}
    if expression.func in functions:
        return functions[expression.func](*args)
    raise TypeError(f"unsupported reference expression: {expression}")


def evaluate(expression: sp.Expr, values: Mapping[sp.Symbol | str, object], *,
             dps: int = 80):
    """Evaluate a scalar expression in an isolated mpmath precision context.

    Decimal strings preserve the intended inputs; Python floats retain their
    existing binary approximation. Exact numeric SymPy expressions (e.g. pi/2)
    are also accepted as input values. All supplied values must be finite real
    scalars, and canonical parameter domains are validated before evaluation.
    The returned mpf/mpc keeps its private context; global mpmath.mp.dps is untouched.
    Precision convergence must be checked at the parameter points under study.
    """
    if isinstance(dps, bool) or not isinstance(dps, int) or dps < 15:
        raise ValueError("dps must be an integer of at least 15 decimal digits")
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    bound = {}
    for key, value in values.items():
        symbol = SYMBOLS[key] if isinstance(key, str) else key
        if not isinstance(symbol, sp.Symbol):
            raise TypeError("input keys must be SymPy symbols or canonical symbol names")
        if symbol in bound:
            raise ValueError(f"duplicate input for {symbol}")
        if isinstance(value, bool):
            raise TypeError(f"{symbol} must be a finite real scalar")
        try:
            number = _walk(value, {}, ctx) if isinstance(value, sp.Basic) else ctx.mpf(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{symbol} must be a finite real scalar") from exc
        if isinstance(number, ctx.mpc) or not ctx.isfinite(number):
            raise ValueError(f"{symbol} must be a finite real scalar")
        if symbol in (V, F, M_SEC, D_SEC, F_REF) and number <= 0:
            raise ValueError(f"{symbol} must be positive")
        if symbol == ETA and not 0 < number <= ctx.mpf(1) / 4:
            raise ValueError("eta must satisfy 0 < eta <= 1/4")
        if symbol == IOTA and not 0 <= number <= ctx.pi:
            raise ValueError("iota must satisfy 0 <= iota <= pi")
        bound[symbol] = number
    expression = sp.sympify(expression)
    missing = expression.free_symbols - bound.keys()
    if missing:
        raise ValueError(f"missing reference inputs: {', '.join(sorted(map(str, missing)))}")
    return _walk(expression, bound, ctx)


def evaluate_phase(v, eta, *, dps: int = 80):
    """Evaluate the dimensionless intrinsic phase; inputs may be decimal strings."""
    return evaluate(phase_expr(), {V: v, ETA: eta}, dps=dps)


def evaluate_coefficients(eta, *, dps: int = 80) -> dict:
    """Evaluate every exact PN coefficient independently of the production code."""
    return {name: evaluate(expr, {ETA: eta}, dps=dps)
            for name, expr in coefficients().items()}


def evaluate_waveform(*, frequency, mass_seconds, eta, distance_seconds,
                      inclination="0", tc="0", phic="0",
                      reference_frequency=None, dps: int = 80) -> dict:
    """Scalar restricted waveform in geometric seconds and the adapter convention.

    ``reference_frequency=None`` selects coalescence phase. A strictly positive
    value selects finite-reference subtraction; zero is invalid, not a sentinel.
    Unsupported physics options are absent from the API and are not ignored.
    """
    values = {F: frequency, M_SEC: mass_seconds, ETA: eta, D_SEC: distance_seconds,
              IOTA: inclination, TC: tc, PHIC: phic}
    if reference_frequency is not None:
        values[F_REF] = reference_frequency
    expressions = waveform_expr(reference_frequency=reference_frequency is not None)
    return {name: evaluate(expr, values, dps=dps) for name, expr in expressions.items()}
