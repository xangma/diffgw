"""Paper-math checks of the nonspinning NRTidalv2 phase slice, CPU FP64."""

import importlib
import math
import os

import numpy as np
import pytest

def require_module(name):
    if os.environ.get("DIFFGW_REQUIRE_LAL_BINARY") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


sp = require_module("sympy")
require_module("mpmath")

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import nrtidalv2_phase as ref


# Predeclared cases and thresholds: docs/math/nrtidalv2-phase.md.
CASES = (
    ("equal", 1.35, 1.35, 400.0, 400.0),
    ("unequal", 1.6, 1.2, 200.0, 800.0),
    ("exchanged", 1.2, 1.6, 800.0, 200.0),
    ("zero_tide", 1.35, 1.35, 0.0, 0.0),
)
FREQUENCIES = np.array([30.0, 100.0, 300.0, 500.0, 800.0])
PHASE_ATOL = 1e-10
PHASE_RTOL = 1e-11
DERIVATIVE_ATOL = 1e-10  # rad / Hz
DERIVATIVE_RTOL = 1e-8


def _unit_ratio(backend, frequencies, case):
    _, m1, m2, lambda1, lambda2 = case
    options = dict(mass1=m1, mass2=m2, chi1=0.0, chi2=0.0,
                   distance=100.0, inclination=0.4, phic=0.0, tc=0.0,
                   f_ref=30.0, phasing_convention="reference_frequency")
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = torch.as_tensor(frequencies, dtype=torch.float64)
        hp_bbh, _ = IMRPhenomD()(f=f, **options)
        hp_bns, _ = IMRPhenomD_NRTidalv2()(f=f, lambda1=lambda1,
                                            lambda2=lambda2,
                                            apply_merger_taper=False, **options)
        quotient = hp_bns / hp_bbh
        assert quotient.dtype == torch.complex128
        return quotient / torch.abs(quotient)
    if backend == "jax":
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        hp_bbh, _ = IMRPhenomD.evaluate(f=f, **options)
        hp_bns, _ = IMRPhenomD_NRTidalv2.evaluate(
            f=f, lambda1=lambda1, lambda2=lambda2,
            apply_merger_taper=False, **options)
        quotient = hp_bns / hp_bbh
        assert quotient.dtype == jnp.complex128
        return quotient / jnp.abs(quotient)
    raise ValueError(backend)


def _paper_phase(case, frequency, *, derivative=False):
    _, m1, m2, lambda1, lambda2 = case
    return float(ref.evaluate(frequency, (m1 + m2) * MSUN_SEC,
                              (m1 - m2) / (m1 + m2), lambda1, lambda2,
                              derivative=derivative))


def test_pade_has_printed_post_newtonian_expansion():
    z = sp.symbols("z", positive=True)
    series = sp.series(ref.pade_expr(z**2), z, 0, 6).removeO().expand()
    for order, coefficient in ((0, 1), (2, ref.C1), (3, ref.C32),
                               (4, ref.C2), (5, ref.C52)):
        assert sp.simplify(series.coeff(z, order) - coefficient) == 0
    assert ref.N52 == sp.Rational("90.550822")
    assert ref.N3 == sp.Rational("-60.253578")
    assert ref.D1 == sp.Rational("-15.111208")
    assert ref.D2 == sp.Rational("8.0641096")


def test_paper_body_exchange_and_zero_tides():
    for frequency in FREQUENCIES:
        assert abs(_paper_phase(CASES[1], frequency)
                   - _paper_phase(CASES[2], frequency)) < 1e-12
        assert _paper_phase(CASES[3], frequency) == 0.0


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_frontend_complex_phase_against_paper(backend, case):
    if backend == "jax":
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
    actual = np.asarray(_unit_ratio(backend, FREQUENCIES, case))
    phase = np.array([_paper_phase(case, f) for f in FREQUENCIES])
    expected = np.exp(-1j * phase)
    residual = np.abs(actual - expected)
    assert np.all(residual <= PHASE_ATOL + PHASE_RTOL * np.abs(phase)), (
        backend, case[0], residual.max())


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES[:2], ids=[case[0] for case in CASES[:2]])
def test_frontend_frequency_derivative_against_paper(backend, case):
    frequency = 300.0  # The principal phase branch is smooth at this point.
    expected = _paper_phase(case, frequency, derivative=True)
    if backend == "torch":
        torch = require_module("torch")
        f = torch.tensor(frequency, dtype=torch.float64, requires_grad=True)
        phase = -torch.angle(_unit_ratio(backend, f, case))
        actual = float(torch.autograd.grad(phase, f)[0])
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        actual = float(jax.grad(lambda f: -jnp.angle(
            _unit_ratio(backend, f, case)))(jnp.float64(frequency)))
    assert math.isclose(actual, expected, rel_tol=DERIVATIVE_RTOL,
                        abs_tol=DERIVATIVE_ATOL), (backend, case[0], actual, expected)
