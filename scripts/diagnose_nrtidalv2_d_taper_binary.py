"""Black-box NRTidalv2 D taper diagnostic against the printed public fit.

Run with PYTHONPATH=src:tests using a LALSuite binary installation. The
per-bin merger-boundary inversion is diagnostic only; it is not a fitted
coefficient or an implementation prescription.
"""

import numpy as np
from scipy.optimize import brentq

import lal
import lalsimulation as lalsim

from test_nrtidalv2_amplitude_math_reference import (
    DISTANCE_MPC, INCLINATION, native, paper_merger,
)


def window(f, fm):
    z = (f - fm) / (0.2 * fm)
    if z <= 0:
        return 1.0
    if z >= 1:
        return 0.0
    s = 1 / z + 1 / (z - 1)
    if s < -700:
        return 0.0
    if s > 700:
        return 1.0
    return 1 / (1 + np.exp(-s))


def binary_ratio(case, frequencies):
    _, m1, m2, l1, l2 = case
    grid = lal.CreateREAL8Vector(len(frequencies))
    grid.data[:] = frequencies
    common = (0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0,
              DISTANCE_MPC * 1e6 * lal.PC_SI, INCLINATION)

    def amplitude(name, lam1, lam2):
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, lam1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, lam2)
        hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
            *common, params, lalsim.GetApproximantFromString(name), grid)
        return np.abs(np.asarray(hp.data.data))

    return amplitude("IMRPhenomD_NRTidalv2", l1, l2) / amplitude("IMRPhenomD", 0, 0)


def main():
    print(f"lal={lal.__version__} lalsimulation={lalsim.__version__}",
          flush=True)
    cases = [(f"equal_{lam}", 1.35, 1.35, lam, lam)
             for lam in (0, 100, 200, 400, 800, 1200)]
    cases.append(("unequal_200_800", 1.6, 1.2, 200, 800))
    for case in cases:
        diagnose(case)


def diagnose(case):
    fm = paper_merger(case)
    fractions = np.linspace(0.9, 1.2, 151)
    frequencies = fm * fractions
    ratio_binary = binary_ratio(case, frequencies)
    bbh, bns_untapered = native(case, frequencies, "torch", taper=False)
    ratio_untapered = np.abs(bns_untapered) / np.abs(bbh)
    effective_window = ratio_binary / ratio_untapered
    paper_window = np.array([window(f, fm) for f in frequencies])
    pre = fractions <= 0.99
    mid = (fractions >= 1.025) & (fractions <= 1.175)
    sample = mid & (effective_window > 0.02) & (effective_window < 0.98)
    inferred = np.array([
        brentq(lambda candidate: window(f, candidate) - w,
               fm * 0.998, fm * 1.002, xtol=1e-10)
        for f, w in zip(frequencies[sample], effective_window[sample])
    ])
    ppm = (inferred / fm - 1) * 1e6
    median_fm = float(np.median(inferred))
    median_window = np.array([window(f, median_fm) for f in frequencies[sample]])
    print(case[0], "fm_hz", f"{fm:.9f}",
          "pre_max", f"{np.max(np.abs(ratio_binary[pre] - ratio_untapered[pre])):.9g}",
          "paper_mid_max", f"{np.max(np.abs(effective_window[mid] - paper_window[mid])):.9g}",
          "ppm_median", f"{np.median(ppm):.4f}",
          "ppm_range", f"[{np.min(ppm):.4f},{np.max(ppm):.4f}]",
          "shifted_mid_max", f"{np.max(np.abs(effective_window[sample] - median_window)):.9g}",
          "n", len(ppm), flush=True)


if __name__ == "__main__":
    main()
