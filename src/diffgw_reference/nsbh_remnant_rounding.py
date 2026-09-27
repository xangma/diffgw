"""Paper-precision interval reference for the NSBH remnant fit.

Only printed coefficients and equations from Jiménez-Forteza et al. (2017)
arXiv:1611.00332v2 and Zappa et al. (2019) arXiv:1903.11622v2 enter here.
Arithmetic uses exact rational bounds, including rational enclosures of square
roots, so interval endpoints do not depend on frontend or FP64 evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isqrt


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("reversed interval")

    @classmethod
    def point(cls, value: int | Fraction) -> Interval:
        v = Fraction(value)
        return cls(v, v)

    def __add__(self, other: Interval | int | Fraction) -> Interval:
        other = _interval(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self) -> Interval:
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: Interval | int | Fraction) -> Interval:
        return self + -_interval(other)

    def __rsub__(self, other: Interval | int | Fraction) -> Interval:
        return _interval(other) - self

    def __mul__(self, other: Interval | int | Fraction) -> Interval:
        other = _interval(other)
        corners = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(corners), max(corners))

    __rmul__ = __mul__

    def __truediv__(self, other: Interval | int | Fraction) -> Interval:
        other = _interval(other)
        if other.lower <= 0 <= other.upper:
            raise ZeroDivisionError("interval denominator contains zero")
        return self * Interval(1 / other.upper, 1 / other.lower)

    def __rtruediv__(self, other: Interval | int | Fraction) -> Interval:
        return _interval(other) / self

    def __pow__(self, power: int) -> Interval:
        if power < 0:
            return Interval.point(1) / self ** -power
        if power == 0:
            return Interval.point(1)
        if power % 2 == 0:
            lo = Fraction(0) if self.lower <= 0 <= self.upper else min(self.lower**power, self.upper**power)
            return Interval(lo, max(self.lower**power, self.upper**power))
        return Interval(self.lower**power, self.upper**power)

    def contains(self, value: float, margin: float = 0.0) -> bool:
        v = Fraction.from_float(value)
        m = Fraction.from_float(margin)
        return self.lower - m <= v <= self.upper + m

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower


def _interval(value: Interval | int | Fraction) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


def printed(text: str) -> Interval:
    """Conventional round-to-nearest enclosure for the last printed digit."""
    number = Decimal(text)
    center = Fraction(number)
    half_unit = Fraction(Decimal(5).scaleb(number.as_tuple().exponent - 1))
    return Interval(center - half_unit, center + half_unit)


def sqrt_integer(value: int) -> Interval:
    """Exact rational lower/upper bounds at 60 decimal places."""
    scale = 10**60
    root = isqrt(value * scale * scale)
    return Interval(Fraction(root, scale), Fraction(root + 1, scale))


# Named directly from the printed tables, with no native extra digits.
SPIN_TABLE = {
    "a2": "3.833", "a3": "-9.49", "a5": "2.513",  # Table 1
    "b1": "1.00096", "b2": "0.788", "b3": "0.654", "b5": "0.840",  # Table 2
    "f21": "8.774", "f31": "22.83", "f50": "1.8805", "f11": "4.4092",  # Table 3
    "d10": "0.322", "d11": "9.33", "d20": "-0.0598",  # Table 4
    "d30": "2.32", "d31": "-3.26", "f12": "0.512",
    "f22": "-32.1", "f32": "-154", "f51": "-4.77",
}
MASS_TABLE = {
    "a2": "0.5610", "a3": "-0.847", "a4": "3.145",  # Table 7
    "b1": "-0.209", "b2": "-0.197", "b3": "-0.159", "b5": "2.985",  # Table 8
    "f20": "4.27", "f30": "31.09", "f50": "1.56735", "f10": "1.81",  # Table 9
    "d10": "-0.098", "d11": "-3.23", "d20": "0.0112",  # Table 10
    "d30": "-0.0198", "d31": "-4.92", "f11": "15.7",
    "f31": "-243.6", "f51": "-0.58",
}
SPIN_SCALES = {
    "a20": "5.24", "a30": "1.3", "a50": "2.88",  # Eq. 7
    "b10": "-0.194", "b20": "0.0851", "b30": "0.00954", "b50": "-0.579",  # Eq. 8
}
MASS_SCALES = {
    "b10": "0.346", "b20": "0.211", "b30": "0.128", "b50": "-0.212",  # Eq. 22
}
SPIN_CONSTRAINT = {
    "f21": "0.345225", "f31": "0.0321306", "f50": "-3.66556", "constant": "7.5397",  # Eq. 13
}
MASS_CONSTRAINT = {
    "f20": "-0.574752", "f30": "-0.280958", "f50": "64.6408", "constant": "-88.3165",  # Eq. 26
}

ZAPPA_MASS_TABLE = (
    ("-1.83e-3", "2.39e-3", "4.29e-3", "9.8e-3"),
    ("2.34e-7", "-8.28e-7", "-1.64e-6", "8.08e-6"),
    ("-2.01e-2", "1.32e-1", "6.51e-2", "-1.43e-1"),
)
ZAPPA_SPIN_TABLE = (
    ("-5.44e-3", "7.91e-3", "2.33e-2", "2.48e-2"),
    ("-8.57e-7", "-2.82e-6", "6.61e-6", "4.29e-5"),
    ("-3.04e-2", "2.55e-1", "1.48e-1", "-4.28e-1"),
)

# Frozen in docs/math/nsbh-remnant-rounding-interval.md before native comparison.
CASES = ((3, "0", 2500), (3, "0.5", 2500), (5, "0.5", 1000),
         (5, "0.75", 100), (5, "0.75", 2500))
FP64_ABSOLUTE_MARGIN = 2e-12


def _coeffs(table: dict[str, str]) -> dict[str, Interval]:
    return {name: printed(value) for name, value in table.items()}


def _constraint(table: dict[str, str], coeffs: dict[str, Interval]) -> Interval:
    result = printed(table["constant"])
    for key, value in table.items():
        if key != "constant":
            result += printed(value) * coeffs[key]
    return result


def _poly_spin(f0: Interval, f1: Interval, f2: Interval, eta: Interval) -> Interval:
    f3 = 64 - 64 * f0 - 16 * f1 - 4 * f2
    return f0 + f1 * eta + f2 * eta**2 + f3 * eta**3


def _poly_mass(f0: Interval, f1: Interval, eta: Interval) -> Interval:
    f2 = 16 - 16 * f0 - 4 * f1
    return f0 + f1 * eta + f2 * eta**2


def bbh_remnant_interval(q: int, chi_bh: str) -> tuple[Interval, Interval]:
    """Jiménez-Forteza final mass ratio and final spin at chi_NS=0."""
    qf = Fraction(q)
    chi = Interval.point(Fraction(Decimal(chi_bh)))
    eta = Interval.point(qf / (qf + 1) ** 2)
    shat = chi * qf**2 / (qf**2 + 1)
    stot = chi * qf**2 / (qf + 1) ** 2
    delta_chi = chi
    root1m4eta = Interval.point(Fraction(q - 1, q + 1))
    sqrt2, sqrt3 = sqrt_integer(2), sqrt_integer(3)

    m, ms = _coeffs(MASS_TABLE), _coeffs(MASS_SCALES)
    f10 = _constraint(MASS_CONSTRAINT, m)
    energy_zero = ((1 - 2 * sqrt2 / 3) * eta + m["a2"] * eta**2
                   + m["a3"] * eta**3 + m["a4"] * eta**4)
    mass_spin_num = 1
    for i, f0, f1 in ((1, f10, m["f11"]), (2, m["f20"], Interval.point(0)),
                      (3, m["f30"], m["f31"])):
        mass_spin_num += ms[f"b{i}0"] * m[f"b{i}"] * shat**i * _poly_mass(f0, f1, eta)
    mass_spin_den = 1 + ms["b50"] * m["b5"] * shat * _poly_mass(m["f50"], m["f51"], eta)
    erad = (energy_zero * mass_spin_num / mass_spin_den
            + m["d10"] * root1m4eta * eta**2 * (1 + m["d11"] * eta) * delta_chi
            + m["d20"] * eta**3 * delta_chi**2
            + m["d30"] * root1m4eta * eta * (1 + m["d31"] * eta) * shat * delta_chi)
    mass_ratio = 1 - erad

    s, ss = _coeffs(SPIN_TABLE), _coeffs(SPIN_SCALES)
    f11 = _constraint(SPIN_CONSTRAINT, s)
    lorb_zero = ((2 * sqrt3 * eta + ss["a20"] * s["a2"] * eta**2
                  + ss["a30"] * s["a3"] * eta**3)
                 / (1 + ss["a50"] * s["a5"] * eta))
    spin_num = 0
    for i, f1, f2 in ((1, f11, s["f12"]), (2, s["f21"], s["f22"]),
                      (3, s["f31"], s["f32"])):
        spin_num += ss[f"b{i}0"] * s[f"b{i}"] * shat**i * _poly_spin(Interval.point(0), f1, f2, eta)
    spin_den = 1 + ss["b50"] * s["b5"] * shat * _poly_spin(s["f50"], s["f51"], Interval.point(0), eta)
    lorb = (lorb_zero + spin_num / spin_den
            + s["d10"] * root1m4eta * eta**2 * (1 + s["d11"] * eta) * delta_chi
            + s["d20"] * eta**3 * delta_chi**2
            + s["d30"] * root1m4eta * eta**3 * (1 + s["d31"] * eta) * shat * delta_chi)
    return mass_ratio, stot + lorb


def zappa_factor_interval(q: int, chi_bh: str, lam: int,
                          table: tuple[tuple[str, ...], ...]) -> Interval:
    """Zappa supplementary Eqs. (4)–(6), without extrapolation guards."""
    eta = Interval.point(Fraction(q, (q + 1) ** 2))
    chi = Interval.point(Fraction(Decimal(chi_bh)))
    p = []
    for row in table:
        c10, c11, c20, c21 = (printed(value) for value in row)
        p.append((c10 * chi + c11) * eta + (c20 * chi + c21) * eta**2)
    return (1 + p[0] * lam + p[1] * lam**2) / (1 + p[2] ** 2 * lam) ** 2


def nsbh_remnant_interval(q: int, chi_bh: str, lam: int) -> tuple[Interval, Interval]:
    mass_ratio, spin = bbh_remnant_interval(q, chi_bh)
    mass_factor = zappa_factor_interval(q, chi_bh, lam, ZAPPA_MASS_TABLE)
    spin_factor = zappa_factor_interval(q, chi_bh, lam, ZAPPA_SPIN_TABLE)
    total_mass = Fraction(7, 5) * (q + 1)
    return total_mass * mass_ratio * mass_factor, spin * spin_factor
