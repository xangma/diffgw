"""Aligned black-hole spin phase corrections through 3.5PN for TaylorF2.

This separately versioned specification composes the unchanged nonspinning
reference with spin-orbit, spin-spin and cubic-spin phase terms. Spins chi1 and
chi2 are signed projections along the orbital angular momentum, with |chi_i|<=1.
The signed mass difference delta=(m1-m2)/(m1+m2) obeys -1<delta<1; consequently
eta=(1-delta**2)/4 and Xi=(1+/-delta)/2. These coordinates are smooth at equal
mass and permit either labelling of the bodies. All symbolic derivatives hold
the other canonical inputs fixed, including delta when differentiating spin.

Wade et al. Eq. (4) and its following coupling definitions supply 1.5--2.5PN
spin terms, with black-hole quadrupole parameters qi=1. Mishra et al. Eqs.
(5)--(7) supply 3PN and 3.5PN spin-orbit, 3PN spin-spin and 3.5PN cubic terms.
Only their dimensionless phase-polynomial coefficients are used: the orbital
phase prefactor printed in Mishra Eq. (4) is inconsistent with its harmonic
construction in Eq. (1). The standard second-harmonic 3/(128*eta*v**5)
normalization, amplitude, Fourier sign and phase gauges are inherited from the
base reference. Spin 2.5PN logarithms use log(v), i.e. Wade's v_ref=1 gauge.

The 3.5PN quadratic-spin coefficient includes the explicit tail flux from
Messina and Nagar Eq. (26), using their corrected arXiv v4. It is obtained by
energy balance, not by treating a flux coefficient as a phase coefficient.
``ss7_balance_terms`` exposes the derivation. Pratten et al. Eq. (A19)'s
quadratic terms equal the result with that explicit tail flux omitted; their
Eq. (A24) separately supplies the additional +20*F7SS. This reference includes
both (A19) and (A24), independently checked by the balance derivation. No
production waveform or external waveform implementation supplies coefficients.

The reference covers the adiabatic flux-to-infinity inspiral with black-hole
spin-induced multipoles. It excludes horizon absorption (which already enters
at 2.5PN), neutron-star multipole deformations, tides, precession, eccentricity,
spin-dependent amplitude corrections, merger, cutoffs and tapering. Thus
'through 3.5PN' describes the included conservative/flux-to-infinity spin phase,
not every physical black-hole effect at those orders. Its analytic domain is
not a guarantee of PN accuracy outside slow inspiral. Terms are chosen by
physical identity, independently of frontend option recipes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import sympy as sp

from . import taylorf2 as base

SPEC_VERSION = "taylorf2-aligned-bh-spin-3.5pn/v1"
REFERENCE_VERSION = SPEC_VERSION
INHERITED_SPEC_VERSION = base.SPEC_VERSION
INHERITED_COMPONENT_IDS = dict(base.COMPONENT_IDS)

V, F, M_SEC, D_SEC, F_REF = base.V, base.F, base.M_SEC, base.D_SEC, base.F_REF
IOTA, TC, PHIC = base.IOTA, base.TC, base.PHIC
DELTA, CHI1, CHI2 = sp.symbols("delta chi1 chi2", real=True)
ETA_EXPR = (1 - DELTA**2) / 4
SYMBOLS = {s.name: s for s in
           (V, F, M_SEC, D_SEC, F_REF, IOTA, TC, PHIC, DELTA, CHI1, CHI2)}
SPIN_TERMS = ("SO3", "SS4", "SO5", "SO6", "SS6", "SO7", "SS7", "SSS7")

SOURCES = {
    **base.SOURCES,
    "wade2013": {
        "arxiv": "1306.3901v2",
        "url": "https://arxiv.org/pdf/1306.3901v2",
        "doi": "10.1103/PhysRevD.88.083002",
        "equations": ("4",),
        "use": "Phase polynomial and beta, sigma, gamma definitions below Eq. (4), PDF p. 4; qi=1.",
    },
    "mishra2016": {
        "arxiv": "1601.05588v1",
        "url": "https://arxiv.org/pdf/1601.05588v1",
        "doi": "10.1103/PhysRevD.93.084054",
        "equations": ("1", "4", "5", "6a", "6b", "7"),
        "use": "Spin polynomial only; orbital normalization caveat in module documentation.",
    },
    "bohe2015": {
        "arxiv": "1501.01529v1",
        "url": "https://arxiv.org/pdf/1501.01529v1",
        "doi": "10.1088/0264-9381/32/19/195010",
        "equations": ("3.33", "4.14"),
        "use": "Leading aligned BH spin-spin energy and flux for the SS7 balance derivation.",
    },
    "messina2017": {
        "arxiv": "1703.08107v4",
        "url": "https://arxiv.org/pdf/1703.08107v4",
        "doi": "10.1103/PhysRevD.95.124001",
        "equations": ("23", "26", "27"),
        "errata": ("10.1103/PhysRevD.96.049907", "10.1103/PhysRevD.97.109902"),
        "use": "3.5PN SS tail flux; v4 incorporates the published errata.",
    },
    "pratten2020": {
        "arxiv": "2001.11412v1",
        "url": "https://arxiv.org/pdf/2001.11412v1",
        "doi": "10.1103/PhysRevD.102.064001",
        "equations": ("A19", "A23", "A24"),
        "use": "Quadratic phase in A19 plus the explicit SS tail increment in A24; A23 gives its flux.",
    },
    "buonanno2009-balance": {
        **base.SOURCES["buonanno2009"],
        "equations": ("3.2", "3.3a", "3.3b", "3.14"),
        "use": "F3NS=4*pi, energy balance and second-harmonic SPA for SS7 derivation.",
    },
}

COMPONENT_IDS = {
    name: f"tf2.bhspin.{name}.v1"
    for name in (*SPIN_TERMS, "SO5_log", "intrinsic_phase", "coalescence_phase",
                 "reference_phase", "polarizations")
}
COMPONENT_SOURCES = {
    COMPONENT_IDS[name]: ("wade2013", "4")
    for name in ("SO3", "SS4", "SO5", "SO5_log")
}
COMPONENT_SOURCES.update({
    COMPONENT_IDS["SO6"]: ("mishra2016", "5", "6a", "7"),
    COMPONENT_IDS["SS6"]: ("mishra2016", "5", "6a", "7"),
    COMPONENT_IDS["SO7"]: ("mishra2016", "5", "6b", "7"),
    COMPONENT_IDS["SSS7"]: ("mishra2016", "5", "6b", "7"),
    COMPONENT_IDS["SS7"]: ("pratten2020", "A19", "A24"),
    **{COMPONENT_IDS[name]: ("project-convention", SPEC_VERSION)
       for name in ("intrinsic_phase", "coalescence_phase", "reference_phase", "polarizations")},
})
DERIVATIONS = {
    COMPONENT_IDS["SS7"]: {
        "sources": (("bohe2015", "3.33", "4.14"),
                    ("messina2017", "23", "26"),
                    ("buonanno2009-balance", "3.2", "3.3a", "3.3b", "3.14")),
        "expression": "ss7_balance_terms",
        "normalization": "E=-eta*M*v^2/2*(1+E4SS*v^4+...); F=32*eta^2*v^10/5*(1+...).",
        "balance": "T7SS=2*F3NS*F4SS-3*E4SS*F3NS-F7SS in dt/dv=5*M/(32*eta)*v^-9*(1+...).",
        "spa": "P_k=40*T_k/((k-5)*(k-8)); k=7 gives SS7=-20*T7SS.",
        "comparison": "Pratten A19 quadratic part equals SS7_without_tail; A24 equals SS7_tail=20*F7SS.",
    },
}


def _parameters(delta, chi1, chi2):
    delta, chi1, chi2 = map(sp.sympify, (delta, chi1, chi2))
    eta = (1 - delta**2) / 4
    return delta, chi1, chi2, eta, (1 + delta) / 2, (1 - delta) / 2


def ss7_balance_terms(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                      chi2: sp.Expr = CHI2) -> dict[str, sp.Expr]:
    """Exact energy/flux-to-phase derivation of the 3.5PN quadratic term.

    Coefficients exclude powers of v and their Newtonian prefactors. With
    a_i=Xi*chi_i, Bohé's S/(G*M**2)=X1*a1+X2*a2 and
    Sigma/(G*M**2)=a2-a1; set kappa_plus=2, kappa_minus=0 in Eqs. (3.33)
    and (4.14). Expanding -E'/F supplies T7SS. Integrating t(v) and the
    second-harmonic orbital phase gives the generic P_k factor in DERIVATIONS.
    ``SS7_tail`` is the addition to Pratten Eq. (A19), not the full SS7.
    """
    _, chi1, chi2, _, x1, x2 = _parameters(delta, chi1, chi2)
    a1, a2 = x1 * chi1, x2 * chi2
    squares, product = a1**2 + a2**2, a1 * a2
    e4 = -(a1 + a2)**2
    f3 = 4 * sp.pi
    f4 = sp.Rational(33, 16) * squares + sp.Rational(31, 8) * product
    f7 = sp.pi * (sp.Rational(65, 8) * squares + sp.Rational(63, 4) * product)
    without_tail = -20 * (2 * f3 * f4 - 3 * e4 * f3)
    t7 = 2 * f3 * f4 - 3 * e4 * f3 - f7
    return {"E4SS": e4, "F3NS": f3, "F4SS": f4, "F7SS": f7,
            "T7SS": t7, "SS7_without_tail": without_tail, "SS7_tail": 20 * f7,
            "SS7": -sp.pi * (sp.Rational(815, 2) * squares + 785 * product)}


def spin_corrections(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                     chi2: sp.Expr = CHI2) -> dict[str, sp.Expr]:
    """Physical spin coefficients inside 3/(128*eta*v**5), without v powers.

    SO/SS/SSS denote terms linear/quadratic/cubic in spin; the integer denotes
    the power of v. SO5_log multiplies v**5*log(v) and accompanies SO5.
    """
    delta, chi1, chi2, eta, x1, x2 = _parameters(delta, chi1, chi2)
    r = sp.Rational
    chis, chia = (chi1 + chi2) / 2, (chi1 - chi2) / 2
    beta = sum((r(113, 12) * x**2 + r(25, 4) * eta) * chi
               for x, chi in ((x1, chi1), (x2, chi2)))
    sigma = (r(79, 8) * eta * chi1 * chi2
             + r(81, 16) * (x1**2 * chi1**2 + x2**2 * chi2**2))
    gamma = sum(((r(732985, 2268) + r(140, 9) * eta) * x**2
                 + eta * (r(13915, 84) - r(10, 3) * eta)) * chi
                for x, chi in ((x1, chi1), (x2, chi2)))
    return {
        "SO3": 4 * beta,
        "SS4": -10 * sigma,
        "SO5": -gamma,
        "SO6": sp.pi * (r(2270, 3) * delta * chia
                         + (r(2270, 3) - 520 * eta) * chis),
        "SS6": ((r(75515, 144) - r(8225, 18) * eta) * delta * chia * chis
                 + (r(75515, 288) - r(263245, 252) * eta - 480 * eta**2) * chia**2
                 + (r(75515, 288) - r(232415, 504) * eta + r(1255, 9) * eta**2) * chis**2),
        "SO7": ((-r(25150083775, 3048192) + r(26804935, 6048) * eta
                  - r(1985, 48) * eta**2) * delta * chia
                 + (-r(25150083775, 3048192) + r(10566655595, 762048) * eta
                    - r(1042165, 3024) * eta**2 + r(5345, 36) * eta**3) * chis),
        "SS7": ss7_balance_terms(delta, chi1, chi2)["SS7"],
        "SSS7": ((r(14585, 24) - 2380 * eta) * delta * chia**3
                  + (r(14585, 24) - r(475, 6) * eta + r(100, 3) * eta**2) * chis**3
                  + (r(14585, 8) - r(215, 2) * eta) * delta * chia * chis**2
                  + (r(14585, 8) - 7270 * eta + 80 * eta**2) * chia**2 * chis),
        "SO5_log": -3 * gamma,
    }


def _selected_terms(terms: Iterable[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        raise TypeError("terms must be an iterable of physical term names, not a string")
    selected = tuple(terms)
    if any(name not in SPIN_TERMS for name in selected):
        raise ValueError(f"terms must contain only {SPIN_TERMS}")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate physical spin terms are not allowed")
    return tuple(name for name in SPIN_TERMS if name in selected)


def coefficients(delta: sp.Expr = DELTA, chi1: sp.Expr = CHI1,
                 chi2: sp.Expr = CHI2, *,
                 terms: Iterable[str] = SPIN_TERMS) -> dict[str, sp.Expr]:
    """Compose unchanged nonspinning coefficients with selected spin terms."""
    selected = _selected_terms(terms)
    delta = sp.sympify(delta)
    result = base.coefficients((1 - delta**2) / 4)
    corrections = spin_corrections(delta, chi1, chi2)
    for name in selected:
        result[f"phi{name[-1]}"] += corrections[name]
    if "SO5" in selected:
        result["phi5_log"] += corrections["SO5_log"]
    return result


def spin_phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
                    chi1: sp.Expr = CHI1, chi2: sp.Expr = CHI2, *,
                    terms: Iterable[str] = SPIN_TERMS) -> sp.Expr:
    """Additive intrinsic spin phase in the base reference's log(v) gauge."""
    selected = _selected_terms(terms)
    v, delta = sp.sympify(v), sp.sympify(delta)
    eta = (1 - delta**2) / 4
    corrections = spin_corrections(delta, chi1, chi2)
    polynomial = sum((corrections[name] * v**int(name[-1]) for name in selected), sp.S.Zero)
    if "SO5" in selected:
        polynomial += corrections["SO5_log"] * v**5 * sp.log(v)
    return sp.Rational(3, 128) * polynomial / (eta * v**5)


def phase_expr(v: sp.Expr = V, delta: sp.Expr = DELTA,
               chi1: sp.Expr = CHI1, chi2: sp.Expr = CHI2, *,
               terms: Iterable[str] = SPIN_TERMS) -> sp.Expr:
    """Intrinsic phase composed with the frozen nonspinning specification."""
    delta = sp.sympify(delta)
    return (base.phase_expr(v, (1 - delta**2) / 4)
            + spin_phase_expr(v, delta, chi1, chi2, terms=terms))


def amplitude_expr() -> sp.Expr:
    """Inherited Newtonian restricted amplitude with eta expressed by delta."""
    return base.amplitude_expr().subs(base.ETA, ETA_EXPR)


def waveform_expr(*, reference_frequency: bool = False,
                  terms: Iterable[str] = SPIN_TERMS) -> dict[str, sp.Expr]:
    """Restricted waveform, retaining both explicit base adapter phase gauges."""
    selected = _selected_terms(terms)
    inherited = {name: expr.subs(base.ETA, ETA_EXPR)
                 for name, expr in base.waveform_expr(reference_frequency=reference_frequency).items()}
    correction = spin_phase_expr((sp.pi * M_SEC * F)**sp.Rational(1, 3), terms=selected)
    if reference_frequency:
        correction -= spin_phase_expr((sp.pi * M_SEC * F_REF)**sp.Rational(1, 3), terms=selected)
    phase = inherited["phase"] + correction
    amplitude = inherited["amplitude"]
    carrier = amplitude * sp.exp(-sp.I * phase)
    cosine = sp.cos(IOTA)
    return {"amplitude": amplitude, "phase": phase,
            "h_plus": (1 + cosine**2) * carrier / 2,
            "h_cross": -sp.I * cosine * carrier}


def derivative_expr(expression: sp.Expr, variable: sp.Symbol | str,
                    order: int = 1) -> sp.Expr:
    """Exact partial derivative in the smooth signed-delta coordinates."""
    if isinstance(variable, str):
        variable = SYMBOLS[variable]
    return base.derivative_expr(expression, variable, order)


def evaluate(expression: sp.Expr, values: Mapping[sp.Symbol | str, object], *,
             dps: int = 80):
    """Scalar evaluation by the unchanged base's isolated mpmath evaluator.

    Decimal strings or exact SymPy input values are preferred; floats retain
    their binary approximation. All supplied values, including unused values,
    are validated. Precision convergence is a numerical check, not source or
    physical validation. Global mpmath precision is never modified.
    """
    bound = {}
    for key, value in values.items():
        symbol = SYMBOLS[key] if isinstance(key, str) else key
        if symbol in bound:
            raise ValueError(f"duplicate input for {symbol}")
        if symbol in (DELTA, CHI1, CHI2):
            number = base.evaluate(symbol, {symbol: value}, dps=dps)
            if symbol == DELTA and not -1 < number < 1:
                raise ValueError("delta must satisfy -1 < delta < 1")
            if symbol in (CHI1, CHI2) and not -1 <= number <= 1:
                raise ValueError(f"{symbol} must satisfy -1 <= {symbol} <= 1")
        bound[symbol] = value
    return base.evaluate(expression, bound, dps=dps)


def evaluate_phase(v, delta, chi1, chi2, *, terms: Iterable[str] = SPIN_TERMS,
                   dps: int = 80):
    """Evaluate the intrinsic phase for one scalar binary and velocity."""
    return evaluate(phase_expr(terms=terms), {V: v, DELTA: delta, CHI1: chi1, CHI2: chi2}, dps=dps)


def evaluate_coefficients(delta, chi1, chi2, *, terms: Iterable[str] = SPIN_TERMS,
                          dps: int = 80) -> dict:
    """Evaluate the combined exact coefficients independently of production."""
    values = {DELTA: delta, CHI1: chi1, CHI2: chi2}
    return {name: evaluate(expr, values, dps=dps)
            for name, expr in coefficients(terms=terms).items()}


def evaluate_waveform(*, frequency, mass_seconds, delta, chi1, chi2, distance_seconds,
                      inclination="0", tc="0", phic="0", reference_frequency=None,
                      terms: Iterable[str] = SPIN_TERMS, dps: int = 80) -> dict:
    """Scalar restricted waveform in geometric seconds and inherited conventions.

    ``reference_frequency=None`` selects coalescence phase; a positive value
    selects finite-reference subtraction. No frontend spin_order or calibration
    flags are accepted: callers must specify the intended physical terms.
    """
    values = {F: frequency, M_SEC: mass_seconds, DELTA: delta, CHI1: chi1, CHI2: chi2,
              D_SEC: distance_seconds, IOTA: inclination, TC: tc, PHIC: phic}
    if reference_frequency is not None:
        values[F_REF] = reference_frequency
    expressions = waveform_expr(reference_frequency=reference_frequency is not None, terms=terms)
    return {name: evaluate(expr, values, dps=dps) for name, expr in expressions.items()}
