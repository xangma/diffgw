"""Measure, without relaxing a gate, the pinned binary's v1 taper mismatch.

Run on a machine with the public LALSimulation binary and with src and tests
on PYTHONPATH. This diagnostic is intentionally outside the strict test set.
"""

import numpy as np

from diffgw_reference.nrtidalv1_taper import taper
from test_nrtidalv1_taper_lal_binary import _lal_waveforms
from test_nrtidalv1_taper_math_reference import (
    ALL_CASES, TAPER_FRACTIONS, native_waveforms, paper_merger,
)


BOUNDARY_ABS_MAX = 5e-5
FULL_COMPLEX_L2_MAX = 1e-6


def main():
    for case in ALL_CASES:
        fm = paper_merger(case)
        frequencies = tuple(fm*TAPER_FRACTIONS)
        bbh, bns = _lal_waveforms(case, frequencies)
        assert np.all(np.abs(bbh) > 0)
        observed = np.abs(bns/bbh)
        expected = np.array([float(taper(f, fm)) for f in frequencies])
        error = np.max(np.abs(observed-expected))
        print(f"{case[0]} f_merger_Hz={fm:.9f} "
              f"max_window_abs={error:.12g} "
              f"boundary_gate={BOUNDARY_ABS_MAX:.0e} "
              f"{'PASS' if error <= BOUNDARY_ABS_MAX else 'FAIL'}")

        full_grid = tuple(np.unique(np.r_[np.arange(30.0, 801.0, 5.0),
                                          fm*TAPER_FRACTIONS]))
        _, lal_full = _lal_waveforms(case, full_grid)
        for backend in ("torch", "jax"):
            _, native_full = native_waveforms(case, np.asarray(full_grid),
                                              backend)
            relative_l2 = np.linalg.norm(native_full-lal_full) / np.linalg.norm(lal_full)
            print(f"  {backend} full_tapered_complex_l2={relative_l2:.12g} "
                  f"full_gate={FULL_COMPLEX_L2_MAX:.0e} "
                  f"{'PASS' if relative_l2 <= FULL_COMPLEX_L2_MAX else 'FAIL'}")


if __name__ == "__main__":
    main()
