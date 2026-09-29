"""Frozen, unaligned NRTidalv2 tidal-phase quotient against a LALSuite binary.

Run the qualification gate with DIFFGW_REQUIRE_LAL_BINARY=1. The ordinary
optional-dependency test suite may skip this module when LALSuite is absent.
No LAL source or coefficients are used.
"""

from functools import lru_cache
from importlib.metadata import version
import os

import numpy as np
import pytest

from test_nrtidalv2_phase_math_reference import CASES, _paper_phase, _unit_ratio, require_module


LALSUITE_VERSION = "7.26.15"
FREQUENCIES = np.arange(30.0, 800.0 + 1e-12, 5.0)
COMPLEX_L2_MAX = 1e-6
MAX_PHASE_RAD = 1e-5


def _lal_modules():
    try:
        import lal
        import lalsimulation as lalsim
    except ImportError as exc:
        if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
            pytest.fail(f"required pinned LALSuite binary unavailable: {exc}")
        pytest.skip(f"optional LALSuite binary unavailable: {exc}")
    assert version("lalsuite") == LALSUITE_VERSION
    return lal, lalsim


@lru_cache(maxsize=None)
def _lal_unit_quotient(case):
    lal, lalsim = _lal_modules()
    _, m1, m2, lambda1, lambda2 = case
    grid = lal.CreateREAL8Vector(len(FREQUENCIES))
    grid.data[:] = FREQUENCIES
    common = (0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              30.0, 100.0 * 1e6 * lal.PC_SI, 0.4)

    def plus(approximant, lam1, lam2):
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, lam1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, lam2)
        hp, _ = lalsim.SimInspiralChooseFDWaveformSequence(
            *common, params, lalsim.GetApproximantFromString(approximant), grid)
        return np.asarray(hp.data.data)

    bbh = plus("IMRPhenomD", 0.0, 0.0)
    bns = plus("IMRPhenomD_NRTidalv2", lambda1, lambda2)
    assert np.all(np.abs(bbh) > 0) and np.all(np.abs(bns) > 0)
    quotient = bns / bbh
    return quotient / np.abs(quotient)


def _metrics(actual, expected):
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    complex_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    max_phase = np.max(np.abs(np.angle(actual * np.conj(expected))))
    return float(complex_l2), float(max_phase)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_lal_binary_quotient_against_paper(case):
    paper = np.exp(-1j * np.array([_paper_phase(case, f) for f in FREQUENCIES]))
    actual = _lal_unit_quotient(case)
    complex_l2, max_phase = _metrics(actual, paper)
    print(f"LAL {case[0]} complex_l2={complex_l2:.9g} max_phase_rad={max_phase:.9g}")
    assert complex_l2 <= COMPLEX_L2_MAX
    assert max_phase <= MAX_PHASE_RAD


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_frontend_quotient_against_lal_binary(backend, case):
    if backend == "jax":
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
    actual = np.asarray(_unit_ratio(backend, FREQUENCIES, case))
    expected = _lal_unit_quotient(case)
    complex_l2, max_phase = _metrics(actual, expected)
    print(f"{backend} {case[0]} complex_l2={complex_l2:.9g} max_phase_rad={max_phase:.9g}")
    assert complex_l2 <= COMPLEX_L2_MAX
    assert max_phase <= MAX_PHASE_RAD
