"""Independent scalar classification from Thompson et al. (2020), Table I."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class MergerBranch:
    name: str
    tide_at_or_above_ringdown: bool
    has_torus: bool


def classify_branch(f_tide_hz, f_rd_hz, torus_ratio, *, dps: int = 70) -> MergerBranch:
    """Classify by the printed Table I inequalities, including equality."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    tide, ringdown, torus = (ctx.mpf(str(value)) for value in (f_tide_hz, f_rd_hz, torus_ratio))
    if not all(ctx.isfinite(value) for value in (tide, ringdown, torus)) or tide <= 0 or ringdown <= 0:
        raise ValueError("Expected finite positive frequencies and a finite torus ratio")
    high = bool(tide >= ringdown)
    has_torus = bool(torus > 0)
    names = {
        (True, False): "non_disruptive",
        (True, True): "mildly_disruptive_torus",
        (False, False): "mildly_disruptive_no_torus",
        (False, True): "disruptive",
    }
    return MergerBranch(names[(high, has_torus)], high, has_torus)
