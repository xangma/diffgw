"""Strict expected-failure Eq. (24) amplitude diagnostics, outside green math CI.

The frozen limits remain the qualification criteria. See the measured failure
receipt in docs/validation/nrtidalv2_amplitude_20260926.md.
"""

import math

import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI
from diffgw_reference.nrtidalv2_phase import lambda_tilde_expr
from test_nrtidalv2_phase_math_reference import CASES, require_module
from test_nrtidalv2_amplitude_math_reference import (
    AMP_FREQUENCIES, DISTANCE_MPC, INCLINATION,
    native, paper_amplitude, paper_parameters,
)


SCALAR_REL_MAX = 1e-11
RATIO_ABS_MAX = 1e-10
SCALAR_FAILURE_CASES = [
    pytest.param(
        case, id=case[0],
        marks=pytest.mark.xfail(
            strict=True,
            reason=("Dietrich 2019 Eq. 24 (arxiv:1905.06011v2) "
                    "sqrt(5*pi*eta/24) vs native 1/2; "
                    "observed relative 0.23607725316 equal, 0.24888677449 unequal; "
                    "docs/validation/nrtidalv2_amplitude_20260926.md"),
        ),
    ) if case[0] != "zero_tide" else pytest.param(case, id=case[0])
    for case in CASES
]
WAVEFORM_FAILURE_CASES = [
    pytest.param(
        case, id=case[0],
        marks=pytest.mark.xfail(
            strict=True,
            reason=("Dietrich 2019 Eq. 24 (arxiv:1905.06011v2) "
                    "plus-polarization quotient; observed absolute "
                    "0.01553696317 equal, 0.01795564024 unequal; "
                    "docs/validation/nrtidalv2_amplitude_20260926.md"),
        ),
    ) if case[0] != "zero_tide" else pytest.param(case, id=case[0])
    for case in CASES
]


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", SCALAR_FAILURE_CASES)
def test_native_amplitude_against_paper(backend, case):
    m_sec, delta, _, _ = paper_parameters(case)
    lt = float(lambda_tilde_expr(delta, case[3], case[4]))
    x = (math.pi*m_sec*AMP_FREQUENCIES)**(2/3)
    d_sec = DISTANCE_MPC*MPC_SI/C_SI
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import _nrtidal_amplitude
        array = lambda v: torch.as_tensor(v, dtype=torch.float64)
    else:
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import _nrtidal_amplitude
        array = lambda v: jnp.asarray(v, dtype=jnp.float64)
    amp = np.asarray(_nrtidal_amplitude(array(x), array(m_sec),
                                         array(d_sec), array(lt)))
    expected = np.array([paper_amplitude(case, f) for f in AMP_FREQUENCIES])
    if case[0] == "zero_tide":
        assert np.all(amp == 0)
    else:
        assert np.max(np.abs(amp/expected - 1)) <= SCALAR_REL_MAX


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", WAVEFORM_FAILURE_CASES)
def test_native_waveform_amplitude_against_paper(backend, case):
    bbh, bns = native(case, AMP_FREQUENCIES, backend, taper=False)
    projection = (1+math.cos(INCLINATION)**2)/2
    expected = 1 + projection*np.array(
        [paper_amplitude(case, f) for f in AMP_FREQUENCIES])/np.abs(bbh)
    actual = np.abs(bns)/np.abs(bbh)
    assert np.max(np.abs(actual-expected)) <= RATIO_ABS_MAX
