#!/usr/bin/env python3
"""Reproducible printed-decimal interval check for SEOBNRv4 NQC fits.

This is an audit diagnostic, not runtime torchwave code.  At the requested
point (nu=.24, chi=0), each distinct printed coefficient is independently
perturbed by half its last shown digit.  The expressions are monotone in each
interval variable here, so enumerating interval corners gives the extrema.
"""
import itertools
import json
import math


def corners(values):
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        yield [value + sign * half for (value, half), sign in zip(values, signs)]


def frequency_bounds():
    values = ((10.262073, .5e-6), (25.850378, .5e-6),
              (.562679, .5e-6), (-.087062, .5e-6))
    result = []
    for a, b, f0, f1 in corners(values):
        a3 = a + 4.0 * (a - b) * (.24 - .25)
        result.append(f0 + f1 * math.log(a3))
    return min(result), max(result)


def slope_bounds():
    # At chi=0, only these distinct literals remain in A10-A12.  .004087 is
    # included as a rounded fit coefficient; the paper does not declare it exact.
    values = ((-.011210, .5e-6), (68.474666, .5e-6),
              (.004087, .5e-6), (.011282, .5e-6), (.015743, .5e-6))
    result = []
    for tpl, log_argument, log_coefficient, equal_mass, linear in corners(values):
        tpl += log_coefficient * math.log(log_argument)
        nu = .24
        result.append(tpl + linear * (nu - .001) +
                      (equal_mass - tpl - linear * (.25 - .001)) /
                      (.25 ** 2 - .001 ** 2) * (nu ** 2 - .001 ** 2))
    return min(result), max(result)


def nominal_frequency():
    a3 = 10.262073 + 4.0 * (10.262073 - 25.850378) * (.24 - .25)
    return .562679 - .087062 * math.log(a3)


def nominal_slope():
    tpl = -.011210 + .004087 * math.log(68.474666)
    nu = .24
    return tpl + .015743 * (nu - .001) + (.011282 - tpl - .015743 * (.25 - .001)) / (.25 ** 2 - .001 ** 2) * (nu ** 2 - .001 ** 2)


def record(bounds, nominal, residual):
    lo, hi = bounds
    return {"interval": [lo, hi], "half_width": (hi - lo) / 2.0,
            "nominal": nominal, "distance_below_nominal": nominal - lo,
            "distance_above_nominal": hi - nominal,
            "screening_residual": residual,
            "signed_residuals_within_interval": {
                "positive": nominal + residual <= hi,
                "negative": nominal - residual >= lo}}


if __name__ == "__main__":
    print(json.dumps({
        "point": {"nu": .24, "chi": 0.0},
        "frequency": record(frequency_bounds(), nominal_frequency(), 7.0e-7),
        "frequency_slope": record(slope_bounds(), nominal_slope(), 4.6e-7),
    }, indent=2, sort_keys=True))
