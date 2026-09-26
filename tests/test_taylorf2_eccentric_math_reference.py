"""Paper-derived CPU float64 gate for the TaylorF2 eccentric phase.

This checks Moore et al. Eq. (6.26) in isolation and the project's separate
f_ref phase gauge. It does not qualify the optional amplitude adapter, the
quasicircular carrier, or physical accuracy over the full parameter domain.
"""

import cmath

import numpy as np
import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import taylorf2_eccentric as paper


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device != "cpu":
        pytest.fail("The eccentric phase math-reference gate is qualified on CPU float64 only")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.taylorf2 import TaylorF2 as Circular
        from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric as Eccentric
        yield Eccentric, Circular, lambda x: torch.as_tensor(x, dtype=torch.float64), "torch"
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.taylorf2 import TaylorF2 as Circular
        from jaxwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric as Eccentric
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(jax.devices("cpu")[0]):
                yield Eccentric, Circular, lambda x: jnp.asarray(x, dtype=jnp.float64), "jax"
        finally:
            jax.config.update("jax_enable_x64", previous)


def phase(backend, f, m1, m2, e0, f0, order):
    model, _, array, _ = backend
    result = model.eccentric_phase_correction(
        f=array(f), mass1=array(m1), mass2=array(m2), e0=array(e0),
        f0=array(f0), ecc_order=order,
    )
    assert result.dtype == array(1.0).dtype
    return np.asarray(result)


@pytest.mark.parametrize("m1,m2,e0,f0", (
    (1.4, 1.4, 0.03, 20.0),
    (1.7, 1.2, 0.07, 30.0),
    (19.0, 7.0, 0.02, 25.0),
))
@pytest.mark.parametrize("order", (0, 2, 3, 4, 5, 6))
def test_each_pn_truncation_against_paper_scalar(backend, m1, m2, e0, f0, order):
    frequencies = (f0, 1.35*f0, 2.4*f0)
    expected = [float(paper.evaluate(f, m1, m2, e0, f0, order))
                for f in frequencies]
    np.testing.assert_allclose(phase(backend, frequencies, m1, m2, e0, f0, order),
                               expected, rtol=2e-12, atol=2e-11)


def test_individual_pn_increments_and_eccentricity_scaling(backend):
    m1, m2, f0, f = 1.6, 1.1, 27.0, 54.0
    previous = 0
    for order in (2, 3, 4, 5, 6):
        actual = float(phase(backend, f, m1, m2, 0.08, f0, order)
                       - phase(backend, f, m1, m2, 0.08, f0, previous))
        expected = float(paper.evaluate(f, m1, m2, 0.08, f0, order)
                         - paper.evaluate(f, m1, m2, 0.08, f0, previous))
        np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=3e-12)
        previous = order
    assert float(phase(backend, f, m1, m2, 0.0, f0, 6)) == 0.0
    one = float(phase(backend, f, m1, m2, 0.04, f0, 6))
    two = float(phase(backend, f, m1, m2, 0.08, f0, 6))
    np.testing.assert_allclose(two, 4*one, rtol=3e-15)


@pytest.mark.parametrize("f0,f_ref", ((20.0, 37.0), (37.0, 20.0), (30.0, 30.0)))
def test_project_reference_frequency_gauge_on_both_polarizations(backend, f0, f_ref):
    eccentric, circular, array, name = backend
    f = (23.0, f_ref, 53.0, 82.0)
    common = dict(f=array(f), mass1=array(1.6), mass2=array(1.2),
                  chi1=array(0.03), chi2=array(-0.02), distance=array(150.0),
                  inclination=array(0.7), phic=array(0.3), tc=array(0.001),
                  f_ref=array(f_ref), phasing_convention="reference_frequency",
                  f_isco_cutoff=False)
    if name == "torch":
        hp_e, hc_e = eccentric()(**common, e0=array(0.025), f0_ecc=array(f0))
        hp_c, hc_c = circular()(**common)
    else:
        hp_e, hc_e = eccentric.evaluate(**common, e0=array(0.025), f0_ecc=array(f0))
        hp_c, hc_c = circular.evaluate(**common)
    for index, frequency in enumerate(f):
        expected = cmath.exp(-1j*float(paper.anchored_phase(
            frequency, 1.6, 1.2, 0.025, f0, f_ref)))
        for ecc, circ in ((hp_e, hp_c), (hc_e, hc_c)):
            actual = complex(ecc[index])/complex(circ[index])
            assert abs(actual - expected) < 3e-10
    # The explicit f_ref gauge sets the eccentric correction to zero here.
    assert float(paper.anchored_phase(f_ref, 1.6, 1.2, 0.025, f0, f_ref)) == 0.0
