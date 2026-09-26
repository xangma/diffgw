"""Predeclared NRTidalv2 amplitude/taper gate against a pinned LAL binary."""

from functools import lru_cache
import math

import numpy as np
import pytest

from test_nrtidalv2_phase_math_reference import CASES
from test_nrtidalv2_phase_lal_binary import _lal_modules
from test_nrtidalv2_amplitude_math_reference import (
    AMP_FREQUENCIES, INCLINATION, TAPER_FRACTIONS, DISTANCE_MPC,
    native, paper_amplitude, paper_merger,
)
from diffgw_reference.nrtidalv2_amplitude import taper


# Frozen before the binary comparison. No phase/time alignment or fit.
BINARY_RATIO_ABS_MAX = 5e-5
BINARY_TAPER_ABS_MAX = 5e-5


PAPER_AMPLITUDE_CASES = [
    pytest.param(
        case, id=case[0],
        marks=pytest.mark.xfail(
            strict=True,
            reason=("Dietrich 2019 Eq. 24 amplitude vs pinned LAL quotient: "
                    "observed 0.0678536536 through merger and 0.0722370285 "
                    "after merger at equal mass; "
                    "docs/validation/nrtidalv2_amplitude_20260926.md"),
        ),
    ) if case[0] != "zero_tide" else pytest.param(case, id=case[0])
    for case in CASES
]
TAPER_BINARY_CASES = [
    pytest.param(
        case, id=case[0],
        marks=pytest.mark.xfail(
            strict=True,
            reason=("Dietrich 2019 Eq. 25 merger taper native/LAL quotient: "
                    "observed 0.000153757344 equal, 0.000189164604 unequal; "
                    "docs/validation/nrtidalv2_amplitude_20260926.md"),
        ),
    ) if case[0] != "zero_tide" else pytest.param(case, id=case[0])
    for case in CASES
]


def binary_frequencies(case):
    return np.unique(np.sort(np.concatenate((
        AMP_FREQUENCIES, paper_merger(case)*TAPER_FRACTIONS))))


@lru_cache(maxsize=None)
def lal_amplitudes(case):
    lal, lalsim = _lal_modules()
    _, m1, m2, lam1, lam2 = case
    frequencies = binary_frequencies(case)
    grid = lal.CreateREAL8Vector(len(frequencies))
    grid.data[:] = frequencies
    common = (0.0, m1*lal.MSUN_SI, m2*lal.MSUN_SI,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              30.0, DISTANCE_MPC*1e6*lal.PC_SI, INCLINATION)

    def plus(name, l1, l2):
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, l1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, l2)
        hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
            *common, params, lalsim.GetApproximantFromString(name), grid)
        return np.abs(np.asarray(hp.data.data))

    bbh = plus("IMRPhenomD", 0.0, 0.0)
    bns = plus("IMRPhenomD_NRTidalv2", lam1, lam2)
    assert np.all(bbh > 0) and np.all(np.isfinite(bns))
    return frequencies, bbh, bns


def paper_ratio(case, frequencies, bbh):
    projection = (1+math.cos(INCLINATION)**2)/2
    f_merger = paper_merger(case)
    correction = np.array([paper_amplitude(case, f) for f in frequencies])
    window = np.array([float(taper(f, f_merger)) for f in frequencies])
    return (1 + projection*correction/bbh)*window


@pytest.mark.parametrize("case", PAPER_AMPLITUDE_CASES)
@pytest.mark.parametrize("region", ("amplitude", "taper"))
def test_lal_amplitude_and_taper_against_paper(case, region):
    frequencies, bbh, bns = lal_amplitudes(case)
    actual = bns/bbh
    expected = paper_ratio(case, frequencies, bbh)
    error = np.abs(actual-expected)
    mask = (frequencies <= paper_merger(case) if region == "amplitude"
            else frequencies > paper_merger(case))
    maximum = np.max(error[mask])
    print(f"LAL {case[0]} {region}_max={maximum:.9g}")
    limit = BINARY_RATIO_ABS_MAX if region == "amplitude" else BINARY_TAPER_ABS_MAX
    assert maximum <= limit


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_frontend_pretaper_amplitude_against_lal_binary(backend, case):
    frequencies, bbh_lal, bns_lal = lal_amplitudes(case)
    bbh, bns = native(case, frequencies, backend, taper=True)
    actual = np.abs(bns)/np.abs(bbh)
    expected = bns_lal/bbh_lal
    mask = frequencies <= paper_merger(case)
    error = np.abs(actual[mask]-expected[mask])
    print(f"{backend} {case[0]} pretaper_binary_max={np.max(error):.9g}")
    assert np.max(error) <= BINARY_RATIO_ABS_MAX


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", TAPER_BINARY_CASES)
def test_frontend_taper_amplitude_against_lal_binary(backend, case):
    frequencies, bbh_lal, bns_lal = lal_amplitudes(case)
    bbh, bns = native(case, frequencies, backend, taper=True)
    mask = frequencies > paper_merger(case)
    actual = np.abs(bns[mask])/np.abs(bbh[mask])
    expected = bns_lal[mask]/bbh_lal[mask]
    error = np.abs(actual-expected)
    print(f"{backend} {case[0]} taper_binary_max={np.max(error):.9g}")
    assert np.max(error) <= BINARY_TAPER_ABS_MAX
