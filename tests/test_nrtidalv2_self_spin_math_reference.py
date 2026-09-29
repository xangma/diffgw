"""Strict CPU FP64 check of Dietrich (2019) Eqs. 26–29 and the paper D composition."""

import math

import mpmath as mp
import numpy as np
import pytest

from diffgw_core.constants import MSUN_SEC
from diffgw_reference import nrtidalv2_phase, nrtidalv2_self_spin as ref
from test_nrtidalv2_phase_math_reference import require_module


CASES = (
    ("equal", 1.35, 1.35, 0.20, 0.20, 350.0, 350.0),
    ("unequal", 1.6, 1.2, 0.04, -0.03, 250.0, 800.0),
    ("exchanged", 1.2, 1.6, -0.03, 0.04, 800.0, 250.0),
    ("zero_spin", 1.45, 1.25, 0.0, 0.0, 300.0, 650.0),
)
FREQUENCIES = np.array((30.0, 100.0, 300.0, 800.0))
COMPONENT_ATOL = 1e-12
COMPLEX_ATOL = 1e-10
DERIVATIVE_ATOL = 1e-10


def _x(case, frequency):
    return (math.pi * (case[1] + case[2]) * MSUN_SEC * frequency) ** (2 / 3)


def _reference(case, frequency):
    _, m1, m2, s1, s2, lam1, lam2 = case
    with mp.workdps(60):
        return tuple(float(v) for v in ref.components(
            mp.power(mp.pi * mp.mpf(str((m1 + m2) * MSUN_SEC))
                     * mp.mpf(str(frequency)), mp.mpf(2)/3),
            *map(str, (m1, m2, s1, s2, lam1, lam2))))


def _native_components(backend, case, frequency):
    _, m1, m2, s1, s2, lam1, lam2 = case
    x = _x(case, frequency)
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import _nrtidal_self_spin_components as fn
        def t(value):
            return torch.tensor(value, dtype=torch.float64)
        return tuple(float(v) for v in fn(t(x), t(m1), t(m2), t(s1),
                                           t(s2), t(lam1), t(lam2), fit="dietrich_2019"))
    jax = require_module("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenomd_nrtidal import _nrtidal_self_spin_components as fn
    def a(value):
        return jnp.asarray(value, dtype=jnp.float64)
    return tuple(float(v) for v in fn(a(x), a(m1), a(m2), a(s1),
                                       a(s2), a(lam1), a(lam2), fit="dietrich_2019"))


def _unit_quotient(backend, case, *, reference_frequency):
    _, m1, m2, s1, s2, lam1, lam2 = case
    options = dict(mass1=m1, mass2=m2, chi1=s1, chi2=s2, distance=100.0,
                   inclination=0.4, phic=0.0, tc=0.0,
                   phasing_convention=("reference_frequency" if reference_frequency
                                       else "coalescence"))
    if reference_frequency:
        options["f_ref"] = 30.0
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = torch.as_tensor(FREQUENCIES, dtype=torch.float64)
        bbh, _ = IMRPhenomD()(f=f, **options)
        bns, _ = IMRPhenomD_NRTidalv2(amplitude_variant="paper_2019")(
            f=f, lambda1=lam1, lambda2=lam2, apply_merger_taper=False, **options)
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd import IMRPhenomD
        from jaxwave.approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidalv2
        f = jnp.asarray(FREQUENCIES, dtype=jnp.float64)
        bbh, _ = IMRPhenomD.evaluate(f=f, **options)
        bns, _ = IMRPhenomD_NRTidalv2.evaluate(
            f=f, lambda1=lam1, lambda2=lam2, apply_merger_taper=False,
            amplitude_variant="paper_2019", **options)
    return np.asarray(bns / bbh)


def test_paper_eq28_and_fig7_are_internally_inconsistent():
    # The text beside Fig. 7 quotes (4.30, 7.28) at Lambda=350;
    # direct evaluation of printed Eqs. (28)-(29) instead gives these values.
    with mp.workdps(60):
        cq, co = ref.multipoles("350")
        assert abs(float(cq) - 5.296589849610877) < 1e-12
        assert abs(float(co) - 10.498317446839701) < 1e-11
        assert abs(float(cq) - 4.30) > 0.9
        assert abs(float(co) - 7.28) > 3.0


def test_paper_body_exchange_and_zero_spins():
    for frequency in FREQUENCIES:
        np.testing.assert_allclose(_reference(CASES[1], frequency),
                                   _reference(CASES[2], frequency), atol=1e-14, rtol=1e-13)
        assert _reference(CASES[3], frequency) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_frontend_components_match_paper(backend, case):
    for frequency in FREQUENCIES:
        p2, p3, p35 = _reference(case, frequency)
        s23, s35 = _native_components(backend, case, frequency)
        assert abs(s23 - (p2+p3)) <= COMPONENT_ATOL
        assert abs(s35 - p35) <= COMPONENT_ATOL


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("reference_frequency", (False, True), ids=("coalescence", "fref30"))
def test_paper_d_complex_quotient(backend, case, reference_frequency):
    actual = _unit_quotient(backend, case, reference_frequency=reference_frequency)
    _, m1, m2, _, _, lam1, lam2 = case
    phases = []
    s23_ref = sum(_reference(case, 30.0)[:2]) if reference_frequency else 0.0
    for f in FREQUENCIES:
        tidal = float(nrtidalv2_phase.evaluate(
            f, (m1+m2)*MSUN_SEC, (m1-m2)/(m1+m2), lam1, lam2))
        phases.append(tidal + sum(_reference(case, f)[:2]) - s23_ref)
    expected = np.exp(-1j * np.asarray(phases))
    assert np.max(np.abs(actual-expected)) <= COMPLEX_ATOL
    assert np.max(np.abs(np.abs(actual)-1)) <= 1e-12


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_spin_derivative_matches_high_precision_reference(backend):
    case = CASES[1]
    _, m1, m2, s1, s2, lam1, lam2 = case
    x = _x(case, 300.0)
    with mp.workdps(60):
        expected = float(mp.diff(lambda spin: ref.d_phase(
            str(x), str(m1), str(m2), spin, str(s2), str(lam1), str(lam2)),
            mp.mpf(str(s1))))
    if backend == "torch":
        torch = require_module("torch")
        from torchwave.approximants.imrphenomd_nrtidal import _nrtidal_self_spin_components as fn
        spin = torch.tensor(s1, dtype=torch.float64, requires_grad=True)
        def t(value):
            return torch.tensor(value, dtype=torch.float64)
        s23, _ = fn(t(x), t(m1), t(m2), spin, t(s2), t(lam1), t(lam2),
                    fit="dietrich_2019")
        actual = float(torch.autograd.grad(s23, spin)[0])
    else:
        jax = require_module("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxwave.approximants.imrphenomd_nrtidal import _nrtidal_self_spin_components as fn
        def a(value):
            return jnp.asarray(value, dtype=jnp.float64)
        actual = float(jax.grad(lambda spin: fn(
            a(x), a(m1), a(m2), spin, a(s2), a(lam1), a(lam2),
            fit="dietrich_2019")[0])(a(s1)))
    assert abs(actual-expected) <= DERIVATIVE_ATOL
