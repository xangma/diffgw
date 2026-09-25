"""Paper-derived XAS inspiral/ringdown amplitude checks for Torch and JAX."""

from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomxas_amplitude as ref


def exact(value):
    return sp.Rational(str(value))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("XAS amplitude reference requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xas import (
            IMRPhenomXAS, XASCarrierState, solve_inspiral_residual_collocation,
        )
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        return (
            "torch", IMRPhenomXAS, XASCarrierState,
            solve_inspiral_residual_collocation,
            lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device),
        )
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import (
        IMRPhenomXAS, XASCarrierState, solve_inspiral_residual_collocation,
    )
    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError as exc:
        pytest.fail(f"JAX {platform} unavailable: {exc}")
    return (
        "jax", IMRPhenomXAS, XASCarrierState,
        solve_inspiral_residual_collocation,
        lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.float64), device),
    )


def as_numpy(value, name):
    return value.detach().cpu().numpy() if name == "torch" else np.asarray(value)


def test_paper_pn_coefficients_and_exchange_symmetry():
    coefficients = ref.pn_coefficients()
    assert len(coefficients) == 7
    assert coefficients[0] == 1 and coefficients[1] == 0
    assert sp.simplify(coefficients[2] + sp.Rational(323, 224)
                       - sp.Rational(451, 168) * ref.ETA) == 0
    for coefficient in coefficients:
        swapped = coefficient.xreplace({
            ref.CHI1: ref.CHI2, ref.CHI2: ref.CHI1, ref.DELTA: -ref.DELTA,
        })
        assert sp.simplify(swapped - coefficient) == 0
    assert sp.limit(ref.inspiral_shape(), ref.F, 0, dir="+") == 1
    assert sp.simplify(ref.F**sp.Rational(7, 6)
                       * ref.paper_newtonian_factor()
                       - sp.sqrt(2 * ref.ETA / (3 * sp.pi**sp.Rational(1, 3)))) == 0


@pytest.mark.parametrize("eta,chi1,chi2,f", [
    (0.25, 0.0, 0.0, 0.008),
    (0.16, 0.5, -0.3, 0.015),
    (0.21, -0.4, 0.6, 0.03),
    (0.08, 0.8, -0.7, 0.04),
])
def test_frontend_pn_shape_and_derivative(backend, eta, chi1, chi2, f):
    name, frontend, _, _, array = backend
    delta = float(np.sqrt(1 - 4 * eta))
    symbolic = ref.pn_shape(eta=exact(eta), chi1=exact(chi1),
                            chi2=exact(chi2), delta=exact(delta))
    expected = float(symbolic.subs(ref.F, exact(f)).evalf(40))
    expected_slope = float(sp.diff(symbolic, ref.F).subs(ref.F, exact(f)).evalf(40))
    actual, slope = frontend.pn_amp_func(
        array(eta), array(chi1), array(chi2), array(np.pi * f), array(delta),
        return_both=True,
    )
    np.testing.assert_allclose(float(actual), expected, rtol=8e-14, atol=8e-14)
    np.testing.assert_allclose(float(slope), expected_slope, rtol=3e-13, atol=3e-13)


def test_collocation_endpoint_distinction_and_exact_recovery():
    meco, isco = sp.Rational(1, 40), sp.Rational(1, 20)
    transition = ref.carrier_transition_frequency(meco, isco)
    assert transition == sp.Rational(5, 160)
    assert ref.paper_collocation_nodes(meco) != ref.paper_collocation_nodes(transition)
    original = (sp.Rational(4, 5), -sp.Rational(3, 4), sp.Rational(2, 3))
    nodes = ref.paper_collocation_nodes(meco)
    values = [sum(c * (sp.pi * f)**p for c, p in zip(
        original, (sp.Rational(7, 3), sp.Rational(8, 3), sp.Integer(3))))
        for f in nodes]
    fitted = ref.residual_collocation_coefficients(meco, values)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(fitted, original))


@pytest.mark.parametrize("endpoint,residuals", [
    (0.02, (0.1, 0.2, 0.4)),
    (0.035, (-0.05, 0.15, 0.6)),
    (0.06, (1.1, 1.3, 1.8)),
])
def test_frontend_collocation_solve(backend, endpoint, residuals):
    name, _, _, solve, array = backend
    expected = ref.residual_collocation_coefficients(
        exact(endpoint), tuple(map(exact, residuals)))
    actual = solve(array(endpoint), *(array(r) for r in residuals))
    np.testing.assert_allclose([float(x) for x in actual],
                               [float(x.evalf(40)) for x in expected],
                               rtol=2e-12, atol=2e-12)


def test_ringdown_stationary_peak_and_scale():
    parameters = (sp.Rational(1, 10), sp.Rational(1, 50),
                  sp.Rational(4, 5), sp.Rational(3, 5))
    peak = ref.stationary_ringdown_peak(*parameters)
    amplitude = ref.ringdown_amplitude(f_rd=parameters[0], f_damp=parameters[1],
                                      sigma=parameters[2], lam=parameters[3])
    assert sp.simplify(sp.diff(amplitude, ref.F).subs(ref.F, peak)) == 0
    assert sp.simplify(ref.printed_peak(*parameters) - peak) != 0
    value = sp.Rational(7, 5)
    a_r = ref.ringdown_scale_at_node(peak, value, *parameters)
    assert sp.simplify(amplitude.subs({ref.F: peak, ref.AR: a_r}) - value) == 0


@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (30.0, 30.0, 0.0, 0.0),
    (50.0, 10.0, 0.5, -0.3),
    (45.0, 20.0, 0.35, 0.6),
    (90.0, 30.0, -0.4, -0.6),
])
def test_full_frontend_inspiral_and_ringdown_shapes(backend, m1, m2, chi1, chi2):
    name, _, carrier, _, array = backend
    state = carrier(m1, m2, chi1=chi1, chi2=chi2)
    eta = exact(float(state.eta))
    delta = exact(float(np.sqrt(1 - 4 * float(state.eta))))
    rho = tuple(exact(float(getattr(state, f"rho{i}"))) for i in range(1, 4))
    symbolic_ins = ref.paper_inspiral_amplitude(
        eta=eta, chi1=exact(chi1), chi2=exact(chi2), delta=delta, rho=rho,
    )
    symbolic_mr = ref.carrier_ringdown_shape(
        f_rd=exact(float(state.f_rd)), f_damp=exact(float(state.f_damp)),
        sigma=exact(float(state.width_rd) / float(state.f_damp)),
        lam=exact(float(state.lam)),
        a_r=exact(float(state.a_r)),
    )
    end = float(state.f_trans_amp)
    peak = float(state.f_peak)
    assert 0 < end < peak
    lam = float(state.lam)
    assert 0 < lam < 1
    stationary = ref.stationary_ringdown_peak(
        exact(float(state.f_rd)), exact(float(state.f_damp)),
        exact(float(state.width_rd) / float(state.f_damp)), exact(lam),
    )
    np.testing.assert_allclose(peak, float(stationary.evalf(40)),
                               rtol=2e-13, atol=2e-13)
    cases = (
        (np.asarray([0.3, 0.55, 0.8, 0.99]) * end, symbolic_ins),
        (np.asarray([1.05, 1.2, 1.5, 2.0]) * peak, symbolic_mr),
    )
    for mf, symbolic in cases:
        hp, _ = state.eval_waveform(array(mf / float(state.m_sec)), distance=500.0)
        actual = np.abs(as_numpy(hp, name))
        expected = np.asarray([float(symbolic.subs(ref.F, exact(f)).evalf(40))
                               for f in mf])
        np.testing.assert_allclose(actual / actual[0], expected / expected[0],
                                   rtol=3e-12, atol=3e-12)


def test_component_evidence_and_links():
    assert set(ref.COMPONENT_SOURCES) == set(ref.COMPONENT_IDS.values())
    for source, *locations in ref.COMPONENT_SOURCES.values():
        assert source in ref.SOURCES
        assert set(locations) <= set(ref.SOURCES[source]["equations"])
    doc = (Path(__file__).resolve().parents[1] / "docs" / "math"
           / "imrphenomxas.md").read_text()
    for component in ref.COMPONENT_IDS.values():
        assert component in doc
