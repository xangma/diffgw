"""Strict green v1 premerger carrier and zero-tide taper public-binary gates.

Nonzero-tide taper disagreements are recorded in the qualification receipt,
not declared passing by widening these predeclared limits.
"""

from functools import lru_cache

import numpy as np
import pytest

from test_nrtidalv1_phase_lal_binary import _lal_modules
from test_nrtidalv1_taper_math_reference import (
    ALL_CASES, TAPER_FRACTIONS, native_waveforms, paper_merger,
)
from diffgw_reference.nrtidalv1_taper import factor
from test_nrtidalv1_taper_math_reference import paper_parameters


FULL_COMPLEX_L2_MAX = 1e-8
RATIO_ABS_MAX = 5e-5
ZERO_TIDE_TAPER_ABS_MAX = 5e-5


@lru_cache(maxsize=None)
def _lal_waveforms(case, frequencies):
    lal, lalsim = _lal_modules()
    _, m1, m2, lambda1, lambda2 = case
    grid = lal.CreateREAL8Vector(len(frequencies))
    grid.data[:] = frequencies
    common = (0.0, m1*lal.MSUN_SI, m2*lal.MSUN_SI,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              30.0, 100e6*lal.PC_SI, 0.4)

    def plus(name, l1, l2):
        parameters = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(parameters, l1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(parameters, l2)
        hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
            *common, parameters, lalsim.GetApproximantFromString(name), grid)
        return np.asarray(hp.data.data)

    return (plus("IMRPhenomD", 0.0, 0.0),
            plus("IMRPhenomD_NRTidal", lambda1, lambda2))


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_full_v1_carrier_before_merger_against_lal(backend, case):
    fm = paper_merger(case)
    frequencies = tuple(np.unique(np.r_[np.arange(30.0, 801.0, 5.0),
                                       0.9*fm, fm]))
    _, lal_bns = _lal_waveforms(case, frequencies)
    _, native_bns = native_waveforms(case, np.asarray(frequencies), backend)
    assert np.all(np.abs(lal_bns) > 0)
    relative_l2 = np.linalg.norm(native_bns-lal_bns)/np.linalg.norm(lal_bns)
    print(f"{backend} {case[0]} premerger_full_l2={relative_l2:.9g}")
    assert relative_l2 <= FULL_COMPLEX_L2_MAX


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_pinned_binary_factor_before_merger(case):
    fm = paper_merger(case)
    frequencies = tuple(np.unique(np.r_[np.arange(30.0, 801.0, 5.0),
                                       0.9*fm, fm]))
    lal_bbh, lal_bns = _lal_waveforms(case, frequencies)
    assert np.all(np.abs(lal_bbh) > 0)
    actual = lal_bns/lal_bbh
    expected = np.array([complex(factor(f, *paper_parameters(case),
                                        carrier_rounded=True))
                         for f in frequencies])
    assert np.max(np.abs(actual-expected)) <= RATIO_ABS_MAX


def test_zero_tide_binary_taper_against_paper():
    case = next(case for case in ALL_CASES if case[0] == "zero_tide")
    fm = paper_merger(case)
    frequencies = tuple(fm*TAPER_FRACTIONS)
    lal_bbh, lal_bns = _lal_waveforms(case, frequencies)
    assert np.all(np.abs(lal_bbh) > 0)
    actual = lal_bns/lal_bbh
    expected = np.array([complex(factor(f, *paper_parameters(case),
                                        carrier_rounded=True))
                         for f in frequencies])
    assert np.max(np.abs(actual-expected)) <= ZERO_TIDE_TAPER_ABS_MAX
