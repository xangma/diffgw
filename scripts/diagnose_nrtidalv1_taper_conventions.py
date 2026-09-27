"""Discriminate merger-frequency, taper-shape, and coupling conventions.

Uses only the published Eq. (11)/Planck function and public binary waveforms.
The midpoint inversion is diagnostic; it is not a production coefficient fit.
Run with src and tests on PYTHONPATH and pinned LALSuite installed.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from test_nrtidalv1_taper_lal_binary import _lal_waveforms
from test_nrtidalv1_taper_math_reference import (
    ALL_CASES, paper_merger, paper_parameters,
)


FRACTIONS = np.array([1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175])


def window_z(z):
    """McKechan et al. (2010) Eq. (7), descending interior branch."""
    return expit(1/z + 1/(z-1))


def window_at(frequency, merger):
    z = (frequency/merger-1)/0.2
    return np.where(z <= 0, 1.0, np.where(z >= 1, 0.0,
                                          window_z(np.clip(z, 1e-12, 1-1e-12))))


def inferred_merger(frequency, amplitude):
    z = brentq(lambda value: window_z(value)-amplitude, 1e-10, 1-1e-10)
    return frequency/(1+0.2*z)


def kappa2_over_keff(case):
    _, m1, m2, lam1, lam2 = case
    xa, xb = m1/(m1+m2), m2/(m1+m2)
    kappa2 = 3*(xa**4*xb*lam1 + xb**4*xa*lam2)
    # Effective coupling is the v1 scalar reference's exact symbolic form.
    from diffgw_reference.nrtidalv1_phase import kappa_eff_expr
    _, delta, _, _ = paper_parameters(case)
    keff = float(kappa_eff_expr(delta, lam1, lam2))
    return kappa2, keff


def merger_with_kappa2(case):
    fm_eff = paper_merger(case)
    k2, keff = kappa2_over_keff(case)

    def rational(kappa):
        return ((1+0.03354*kappa+0.00004315*kappa*kappa)
                / (1+0.07542*kappa+0.0002236*kappa*kappa))

    return fm_eff*rational(k2)/rational(keff)


def d2_printed_rounding_band(case):
    """Illustrate the interval from rounding only Eq. (11)'s printed d2."""
    fm = paper_merger(case)
    _, keff = kappa2_over_keff(case)
    denominator = 1 + 0.07542*keff + 0.0002236*keff*keff
    # 2.236e-4 has a half-last-digit interval of 5e-8. This is an
    # uncertainty illustration, not a recovered binary coefficient.
    bound = 5e-8*keff*keff
    return fm*denominator/(denominator+bound), fm*denominator/(denominator-bound)


def main():
    for case in ALL_CASES:
        fm = paper_merger(case)
        frequency = fm*FRACTIONS
        bbh, bns = _lal_waveforms(case, tuple(frequency))
        observed = np.abs(bns/bbh)
        midpoint = inferred_merger(frequency[3], observed[3])
        independently_inferred = np.array([
            inferred_merger(f, w) for f, w in zip(frequency, observed)
        ])
        paper_residual = np.max(np.abs(observed-window_at(frequency, fm)))
        shifted_residual = np.max(np.abs(observed-window_at(frequency, midpoint)))
        k2_merger = merger_with_kappa2(case)
        k2_residual = np.max(np.abs(observed-window_at(frequency, k2_merger)))
        rounding_low, rounding_high = d2_printed_rounding_band(case)
        print(f"{case[0]} paper_Hz={fm:.12f} midpoint_Hz={midpoint:.12f} "
              f"relative_shift={midpoint/fm-1:.12g}")
        print(f"  inferred_spread_Hz={np.ptp(independently_inferred):.12g} "
              f"paper_window_max_abs={paper_residual:.12g} "
              f"midpoint_window_max_abs={shifted_residual:.12g}")
        print(f"  kappa2_Hz={k2_merger:.12f} "
              f"kappa2_window_max_abs={k2_residual:.12g}")
        print(f"  d2_rounding_band_Hz=[{rounding_low:.12f}, "
              f"{rounding_high:.12f}] "
              f"midpoint_in_band={rounding_low <= midpoint <= rounding_high}")


if __name__ == "__main__":
    main()
