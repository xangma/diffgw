"""Dietrich et al. (2017) Eq. (7) against the v1 Torch/JAX phase carrier."""

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
from diffgw_reference import nrtidalv1_phase as ref


CASES = (
    ("equal", 1.35, 1.35, 400.0, 400.0),
    ("unequal", 1.6, 1.2, 200.0, 800.0),
    ("exchanged", 1.2, 1.6, 800.0, 200.0),
    ("zero_tide", 1.35, 1.35, 0.0, 0.0),
)
FREQUENCIES = np.array([30.0, 100.0, 300.0, 500.0, 800.0])


def _phase(case, frequency, *, derivative=False, carrier_rounded=False):
    _, m1, m2, lambda1, lambda2 = case
    return float(ref.evaluate(
        frequency, (m1 + m2) * MSUN_SEC,
        (m1 - m2) / (m1 + m2), lambda1, lambda2,
        derivative=derivative, carrier_rounded=carrier_rounded,
    ))


def _unit_ratio(backend, frequencies, case):
    _, m1, m2, lambda1, lambda2 = case
    options = dict(mass1=m1, mass2=m2, chi1=0.0, chi2=0.0,
                   distance=100.0, inclination=0.4, phic=0.0, tc=0.0,
                   f_ref=30.0, phasing_convention="reference_frequency")
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidal
        f = torch.as_tensor(frequencies, dtype=torch.float64)
        hp_bbh, _ = IMRPhenomD()(f=f, **options)
        hp_bns, _ = IMRPhenomD_NRTidal()(f=f, lambda1=lambda1,
                                         lambda2=lambda2,
                                         apply_merger_taper=False, **options)
        quotient = hp_bns / hp_bbh
        assert quotient.dtype == torch.complex128
        return quotient / torch.abs(quotient)
    if backend == "jax":
        require_module("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidal
        f = jnp.asarray(frequencies, dtype=jnp.float64)
        hp_bbh, _ = IMRPhenomD.evaluate(f=f, **options)
        hp_bns, _ = IMRPhenomD_NRTidal.evaluate(
            f=f, lambda1=lambda1, lambda2=lambda2,
            apply_merger_taper=False, **options)
        quotient = hp_bns / hp_bbh
        assert quotient.dtype == jnp.complex128
        return quotient / jnp.abs(quotient)
    raise ValueError(backend)


def test_paper_mass_weighting_and_exact_low_frequency_limit():
    delta, l1, l2 = ref.DELTA, ref.LAMBDA1, ref.LAMBDA2
    eta = (1 - delta**2) / 4
    lambda_tilde = sp.Rational(8, 13) * (
        (1 + 7*eta - 31*eta**2) * (l1 + l2)
        + delta * (1 + 9*eta - 11*eta**2) * (l1 - l2)
    )
    assert sp.simplify(ref.kappa_eff_expr() - sp.Rational(3, 16)*lambda_tilde) == 0
    z = sp.symbols("z", positive=True)
    series = sp.series(ref.pade_expr(z**2), z, 0, 4).removeO().expand()
    assert sp.simplify(series.coeff(z, 2) - ref.C1) == 0
    assert ref.D1 == ref.N1 - ref.C1
    assert ref.D1 != sp.Rational("-19.924")


def test_paper_body_exchange_and_zero_tides():
    for frequency in FREQUENCIES:
        assert abs(_phase(CASES[1], frequency)
                   - _phase(CASES[2], frequency)) < 1e-12
        assert _phase(CASES[3], frequency) == 0.0


def test_rounded_carrier_is_explicitly_distinct_from_paper_literal():
    # A real, bounded source-to-carrier discrepancy: never call the two exact.
    differences = [abs(_phase(CASES[0], f, carrier_rounded=True)
                       - _phase(CASES[0], f)) for f in FREQUENCIES]
    assert 1e-5 < max(differences) < 2e-5


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_frontend_quotient_against_declared_rounded_carrier(backend, case):
    if backend == "jax":
        require_module("jax").config.update("jax_enable_x64", True)
    actual = np.asarray(_unit_ratio(backend, FREQUENCIES, case))
    phase = np.array([_phase(case, f, carrier_rounded=True)
                      for f in FREQUENCIES])
    residual = np.abs(actual - np.exp(-1j * phase))
    assert residual.max() <= 1e-10, (backend, case[0], residual.max())


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES[:2], ids=[case[0] for case in CASES[:2]])
def test_frontend_frequency_derivative(backend, case):
    expected = _phase(case, 300.0, derivative=True, carrier_rounded=True)
    if backend == "torch":
        torch = require_module("torch")
        f = torch.tensor(300.0, dtype=torch.float64, requires_grad=True)
        phase = -torch.angle(_unit_ratio(backend, f, case))
        actual = float(torch.autograd.grad(phase, f)[0])
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        actual = float(jax.grad(lambda f: -jnp.angle(
            _unit_ratio(backend, f, case)))(jnp.float64(300.0)))
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10), (
        backend, case[0], actual, expected)
