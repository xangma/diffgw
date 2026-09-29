"""Paper merger/taper and composed v1 factor against Torch and JAX."""

import numpy as np
import pytest
import sympy as sp

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import nrtidalv1_phase as phase_ref
from diffgw_reference import nrtidalv1_taper as ref
from diffgw_reference import nrtidalv2_amplitude as shared_ref
from test_nrtidalv1_phase_lal_binary import HOLDOUTS
from test_nrtidalv1_phase_math_reference import CASES, require_module


ALL_CASES = CASES + HOLDOUTS
TAPER_FRACTIONS = np.array([0.9, 1.0, 1.025, 1.05, 1.1, 1.15, 1.195,
                            1.2, 1.25])
MERGER_REL_MAX = 1e-11
FACTOR_ABS_MAX = 1e-10


def paper_parameters(case):
    _, m1, m2, lambda1, lambda2 = case
    return ((m1 + m2) * MSUN_SEC,
            (m1 - m2) / (m1 + m2), lambda1, lambda2)


def paper_merger(case):
    return float(ref.merger_frequency(*paper_parameters(case)))


def native_waveforms(case, frequencies, backend, *, taper_enabled=True):
    _, m1, m2, lambda1, lambda2 = case
    options = dict(mass1=m1, mass2=m2, chi1=0.0, chi2=0.0,
                   distance=100.0, inclination=0.4, phic=0.0, tc=0.0,
                   f_ref=30.0, phasing_convention="reference_frequency")
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidal
        f = torch.as_tensor(frequencies, dtype=torch.float64)
        bbh, _ = IMRPhenomD()(f=f, **options)
        bns, _ = IMRPhenomD_NRTidal()(f=f, lambda1=lambda1,
                                      lambda2=lambda2,
                                      apply_merger_taper=taper_enabled,
                                      **options)
    elif backend == "jax":
        require_module("jax").config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidal
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        bbh, _ = IMRPhenomD.evaluate(f=f, **options)
        bns, _ = IMRPhenomD_NRTidal.evaluate(
            f=f, lambda1=lambda1, lambda2=lambda2,
            apply_merger_taper=taper_enabled, **options)
    else:
        raise ValueError(backend)
    return np.asarray(bbh), np.asarray(bns)


def test_shared_merger_coupling_is_v1_paper_coupling():
    substitutions = {
        shared_ref.DELTA: phase_ref.DELTA,
        shared_ref.LAMBDA1: phase_ref.LAMBDA1,
        shared_ref.LAMBDA2: phase_ref.LAMBDA2,
    }
    assert sp.simplify(shared_ref.KAPPA.subs(substitutions)
                       - phase_ref.kappa_eff_expr()) == 0


@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_paper_merger_and_window_boundaries(case):
    fm = paper_merger(case)
    assert fm > 0.0
    assert float(ref.taper(fm, fm)) == 1.0
    assert float(ref.taper(1.2*fm, fm)) == 0.0
    window = np.array([float(ref.taper(fm*r, fm)) for r in TAPER_FRACTIONS])
    assert np.all((window >= 0.0) & (window <= 1.0))
    assert np.all(np.diff(window) <= 0.0)
    if case[0] == "exchanged":
        assert abs(fm/paper_merger(CASES[1]) - 1.0) <= 1e-14


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_v1_merger_and_composed_factor_against_paper(backend, case):
    _, m1, m2, l1, l2 = case
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import _bns_merger_frequency
        native_merger = float(_bns_merger_frequency(*(
            torch.as_tensor(v, dtype=torch.float64)
            for v in (m1, m2, l1, l2))))
    else:
        require_module("jax").config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import _bns_merger_frequency
        native_merger = float(_bns_merger_frequency(*(
            jnp.asarray(v, dtype=jnp.float64)
            for v in (m1, m2, l1, l2))))
    fm = paper_merger(case)
    assert abs(native_merger/fm - 1.0) <= MERGER_REL_MAX
    frequencies = np.unique(np.r_[30.0, 800.0, fm*TAPER_FRACTIONS])
    bbh, bns = native_waveforms(case, frequencies, backend)
    assert np.all(np.abs(bbh) > 0)
    actual = bns/bbh
    expected = np.array([complex(ref.factor(f, *paper_parameters(case),
                                            carrier_rounded=True))
                         for f in frequencies])
    assert np.max(np.abs(actual-expected)) <= FACTOR_ABS_MAX
