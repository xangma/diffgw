"""Frozen Eq. (24) amplitude checks against the independent scalar reference.

The frozen limits remain the qualification criteria. See the measured
receipt in docs/validation/nrtidalv2_amplitude_20260926.md.
"""

import math

import numpy as np
import pytest

from diffgw_core.constants import C_SI, MPC_SI
from diffgw_reference.nrtidalv2_phase import lambda_tilde_expr
from test_nrtidalv2_phase_math_reference import CASES, require_module
from test_nrtidalv2_amplitude_math_reference import (
    AMP_FREQUENCIES, DISTANCE_MPC,
    native, paper_amplitude, paper_parameters,
)


SCALAR_REL_MAX = 1e-11
RATIO_ABS_MAX = 1e-10


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_native_amplitude_against_paper(backend, case):
    m_sec, delta, _, _ = paper_parameters(case)
    eta = (1 - delta**2) / 4
    lt = float(lambda_tilde_expr(delta, case[3], case[4]))
    x = (math.pi*m_sec*AMP_FREQUENCIES)**(2/3)
    d_sec = DISTANCE_MPC*MPC_SI/C_SI
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import _nrtidal_amplitude
        def array(v):
            return torch.as_tensor(v, dtype=torch.float64)
    else:
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import _nrtidal_amplitude
        def array(v):
            return jnp.asarray(v, dtype=jnp.float64)
    amp = np.asarray(_nrtidal_amplitude(array(x), array(m_sec),
                                         array(d_sec), array(eta), array(lt)))
    expected = np.array([paper_amplitude(case, f) for f in AMP_FREQUENCIES])
    if case[0] == "zero_tide":
        assert np.all(amp == 0)
    else:
        assert np.max(np.abs(amp/expected - 1)) <= SCALAR_REL_MAX


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES[:2], ids=[c[0] for c in CASES[:2]])
def test_historical_binary_scale_is_distinct_from_published_scale(backend, case):
    """Expose the mass-ratio dependence without deriving 1/2 from the paper."""
    m_sec, delta, _, _ = paper_parameters(case)
    eta = (1 - delta**2) / 4
    lt = float(lambda_tilde_expr(delta, case[3], case[4]))
    x = (math.pi*m_sec*AMP_FREQUENCIES)**(2/3)
    d_sec = DISTANCE_MPC*MPC_SI/C_SI
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import (
            _nrtidal_amplitude, _binary_compatible_nrtidal_amplitude)
        array = lambda v: torch.as_tensor(v, dtype=torch.float64)
    else:
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import (
            _nrtidal_amplitude, _binary_compatible_nrtidal_amplitude)
        array = lambda v: jnp.asarray(v, dtype=jnp.float64)
    published = np.asarray(_nrtidal_amplitude(
        array(x), array(m_sec), array(d_sec), array(eta), array(lt)))
    historical = np.asarray(_binary_compatible_nrtidal_amplitude(
        array(x), array(m_sec), array(d_sec), array(lt)))
    expected = 1/(2*math.sqrt(5*math.pi*eta/24))
    assert np.max(np.abs(historical/published - expected)) <= SCALAR_REL_MAX
    assert expected > 1.2


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_published_d_waveform_omits_tidal_amplitude(backend, case):
    bbh, bns = native(case, AMP_FREQUENCIES, backend, taper=False,
                      amplitude_variant="paper_2019")
    actual = np.abs(bns)/np.abs(bbh)
    assert np.max(np.abs(actual-1)) <= RATIO_ABS_MAX
