"""Strict nonspinning, premerger NRTidalv2 complex-quotient holdouts.

The default D variant is compared with the pinned public LALSuite binary;
its historical amplitude normalization is not asserted to be paper-derived.
"""

import numpy as np
import pytest

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import nrtidalv2_phase as phase_ref
from test_nrtidalv2_amplitude_math_reference import native, paper_merger
from test_nrtidalv2_phase_lal_binary import _lal_modules
from test_nrtidalv2_phase_math_reference import require_module


# Fixed before this binary holdout run. These cases are not in the earlier
# NRTidalv2 phase/amplitude gates or the oracle catalog/holdout inventory.
CASES = (
    ("equal_high_tide", 1.25, 1.25, 900.0, 900.0),
    ("unequal_mixed_tide", 1.48, 1.22, 600.0, 250.0),
    ("exchanged_mixed_tide", 1.22, 1.48, 250.0, 600.0),
)
FREQUENCIES = np.arange(30.0, 1200.0 + 1e-12, 10.0)
COMPLEX_L2_MAX = 1e-6
MAX_PHASE_RAD = 1e-5
AMPLITUDE_RATIO_ABS_MAX = 5e-5


def _lal_quotient(case):
    lal, lalsim = _lal_modules()
    _, m1, m2, lambda1, lambda2 = case
    grid = lal.CreateREAL8Vector(len(FREQUENCIES))
    grid.data[:] = FREQUENCIES
    common = (0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              30.0, 100.0 * 1e6 * lal.PC_SI, 0.4)

    def plus(name, lam1, lam2):
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, lam1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, lam2)
        hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
            *common, params, lalsim.GetApproximantFromString(name), grid)
        return np.asarray(hp.data.data)

    bbh = plus("IMRPhenomD", 0.0, 0.0)
    bns = plus("IMRPhenomD_NRTidalv2", lambda1, lambda2)
    assert np.all(np.abs(bbh) > 0) and np.all(np.abs(bns) > 0)
    return bns / bbh


def _metrics(actual, expected):
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    complex_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    max_phase = np.max(np.abs(np.angle(actual * np.conj(expected))))
    max_amplitude = np.max(np.abs(np.abs(actual) - np.abs(expected)))
    return float(complex_l2), float(max_phase), float(max_amplitude)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_full_tidal_quotient_against_pinned_binary(case):
    assert paper_merger(case) > FREQUENCIES[-1]
    binary = _lal_quotient(case)
    m1, m2, lam1, lam2 = case[1:]
    total = m1 + m2
    phase = np.array([
        float(phase_ref.evaluate(f, total * MSUN_SEC,
                                 (m1 - m2) / total, lam1, lam2))
        for f in FREQUENCIES
    ])
    paper_unit = np.exp(-1j * phase)
    _, binary_phase_error, _ = _metrics(binary / np.abs(binary), paper_unit)
    print(f"{case[0]}: f_merger={paper_merger(case):.9g} Hz, "
          f"binary_paper_phase={binary_phase_error:.9g} rad")
    assert binary_phase_error <= MAX_PHASE_RAD

    for backend in ("torch", "jax"):
        require_module(backend)
        bbh, bns = native(case, FREQUENCIES, backend, taper=True)
        assert np.all(np.abs(bbh) > 0) and np.all(np.abs(bns) > 0)
        quotient = bns / bbh
        complex_l2, max_phase, max_amplitude = _metrics(quotient, binary)
        _, paper_phase_error, _ = _metrics(quotient / np.abs(quotient), paper_unit)
        print(f"{case[0]} {backend}: L2={complex_l2:.9g}, "
              f"phase={max_phase:.9g} rad, amplitude={max_amplitude:.9g}, "
              f"paper_phase={paper_phase_error:.9g} rad")
        assert complex_l2 <= COMPLEX_L2_MAX
        assert max_phase <= MAX_PHASE_RAD
        assert max_amplitude <= AMPLITUDE_RATIO_ABS_MAX
        assert paper_phase_error <= MAX_PHASE_RAD
