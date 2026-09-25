"""Paper-derived XAS inspiral phase and both native carrier checks."""

from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomxas_inspiral_phase as ref
from diffgw_reference import imrphenomxas_phase_branches as middle_ref


def exact(value):
    return sp.Rational(str(value))


def numeric(expr, f, *, precision=45):
    return float(expr.subs(ref.F, exact(f)).evalf(precision))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("XAS inspiral reference requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xas import (
            XASCarrierState, XASParameterFits, pn_phase_canonical_torch,
            d_pn_phase_canonical_torch,
        )
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        return ("torch", XASCarrierState, XASParameterFits,
                pn_phase_canonical_torch, d_pn_phase_canonical_torch,
                lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device))
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import (
        XASCarrierState, XASParameterFits, pn_phase_canonical_jax,
        d_pn_phase_canonical_jax,
    )
    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError as exc:
        pytest.fail(f"JAX {platform} unavailable: {exc}")
    return ("jax", XASCarrierState, XASParameterFits,
            pn_phase_canonical_jax, d_pn_phase_canonical_jax,
            lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.float64), device))


def test_printed_pn_tail_and_pseudo_derivative():
    printed = ref.pn_phase(spin_tail=False)
    carrier = ref.pn_phase(spin_tail=True)
    coeff8 = ref.pn_coefficients()[8]
    v = (sp.pi * ref.F)**sp.Rational(1, 3)
    assert sp.simplify(carrier-printed-3*coeff8*v**3/(128*ref.ETA)) == 0
    assert sp.simplify(coeff8.subs({ref.CHI_S: 0, ref.CHI_A: 0})) == 0
    pseudo = ref.pseudo_pn_phase()
    expected = (ref.SIGMA[1]+ref.SIGMA[2]*ref.F**sp.Rational(1, 3)
                +ref.SIGMA[3]*ref.F**sp.Rational(2, 3)+ref.SIGMA[4]*ref.F)/ref.ETA
    assert sp.simplify(sp.diff(pseudo, ref.F)-expected) == 0


def test_printed_and_carrier_node_distinction_and_exact_solve():
    printed = ref.printed_nodes(sp.Rational(1, 25))
    shifted = ref.carrier_nodes(sp.Rational(1, 25))
    assert shifted[0] == printed[0] == sp.Rational(13, 5000)
    assert shifted[-1] == printed[-1] == sp.Rational(51, 1250)
    assert all(sp.simplify(shifted[i]-printed[i]-shifted[0]) == 0 for i in (1, 2))
    nodes = (sp.Rational(1, 64), sp.Rational(1, 27),
             sp.Rational(1, 8), sp.Integer(1))
    coefficients = (sp.Integer(5), sp.Integer(-3), sp.Integer(2), sp.Integer(7))
    values = [sum(a*f**sp.Rational(j, 3) for j, a in enumerate(coefficients))
              for f in nodes]
    assert ref.collocation_coefficients(nodes, values) == coefficients


@pytest.mark.parametrize("eta,delta,chi1,chi2", [
    (0.25, 0.0, 0.0, 0.0),
    (0.25, 0.0, 0.6, 0.6),
    (0.16, 0.6, 0.5, -0.3),
    (0.1875, 0.5, -0.4, -0.6),
])
def test_native_pn_phase_and_analytic_slope(backend, eta, delta, chi1, chi2):
    _, _, _, phase_fn, derivative_fn, array = backend
    s, a = (chi1+chi2)/2, (chi1-chi2)/2
    expression = ref.pn_phase(eta=exact(eta), delta=exact(delta),
                              chi_s=exact(s), chi_a=exact(a))
    slope = sp.diff(expression, ref.F)
    for f in (0.003, 0.01, 0.04):
        args = (array(f), array(eta), array(chi1), array(chi2), array(delta))
        np.testing.assert_allclose(float(phase_fn(*args)), numeric(expression, f),
                                   rtol=2e-12, atol=2e-8)
        np.testing.assert_allclose(float(derivative_fn(*args)), numeric(slope, f),
                                   rtol=5e-12, atol=2e-6)


@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (30.0, 30.0, 0.0, 0.0),
    (50.0, 10.0, 0.5, -0.3),
    (45.0, 20.0, 0.35, 0.6),
    (90.0, 30.0, -0.4, -0.6),
])
def test_native_collocation_and_inspiral_join(backend, m1, m2, chi1, chi2):
    name, carrier, fits, _, _, array = backend
    state = carrier(m1, m2, chi1=chi1, chi2=chi2)
    eta, s1, s2 = (getattr(state, key) for key in ("eta", "s1_eff", "s2_eff"))
    f_meco = float(fits.fit_f_meco(eta, s1, s2))
    nodes = ref.carrier_nodes(exact(f_meco))
    v3 = float(fits.fit_ins_phase_v3_104(eta, s1, s2))
    d13 = float(fits.fit_ins_phase_d13_104(eta, s1, s2))
    d23 = float(fits.fit_ins_phase_d23_104(eta, s1, s2))
    d43 = float(fits.fit_ins_phase_d43_104(eta, s1, s2))
    values = tuple(map(exact, (v3+d13, v3+d23, v3, v3+d43)))
    recovered = ref.carrier_sigma_from_fits(nodes, values)
    for i, result in enumerate(recovered[1:], start=2):
        np.testing.assert_allclose(float(getattr(state, f"sig{i}")),
                                   float(result.evalf(40)), rtol=5e-9, atol=1e-7)

    e = exact(float(eta))
    delta = exact(float(np.sqrt(max(0, 1-4*float(eta)))))
    chi_s = exact((float(s1)+float(s2))/2)
    chi_a = exact((float(s1)-float(s2))/2)
    pn = ref.pn_phase(eta=e, delta=delta, chi_s=chi_s, chi_a=chi_a)
    middle = middle_ref.intermediate_primitive(
        eta=e, f_rd=exact(float(state.f_rd)), f_damp=exact(float(state.f_damp)),
        coefficients=tuple(exact(float(getattr(state, f"b{i}_val"))) for i in range(5)),
        c0=exact(float(state.c0_val)), a_phi=1, carrier_sign=True)
    middle = middle.subs(middle_ref.F, ref.F)
    join = exact(float(state.f_trans_phase))
    sigmas = tuple(exact(float(getattr(state, f"sig{i}"))) for i in range(2, 5))
    s0, s1_join = ref.join_coefficients(join, e, pn, sigmas, middle)
    np.testing.assert_allclose(float(state.sig1), float(s1_join.evalf(40)),
                               rtol=2e-10, atol=2e-6)
    np.testing.assert_allclose(float(state.sig0), float(s0.evalf(40)),
                               rtol=2e-10, atol=2e-6)
    np.testing.assert_allclose(float(state.sig1), float(recovered[0].evalf(40)),
                               rtol=5e-8, atol=5e-5)
    phase = pn + ref.pseudo_pn_phase(ref.F, e, (s0, s1_join, *sigmas))
    assert abs(numeric(phase-middle, float(join))) < 2e-6
    assert abs(numeric(sp.diff(phase-middle, ref.F), float(join))) < 2e-4

    frequencies = np.linspace(float(join)*0.6, float(join)*0.99, 4)
    hz = frequencies / float(state.m_sec)
    native = state.eval_phase(array(hz))
    actual = native.detach().cpu().numpy() if name == "torch" else np.asarray(native)
    actual = actual - float(state.tshift_sec)*hz - 3*np.pi/4
    expected = np.asarray([numeric(phase, f) for f in frequencies])
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-6)


def test_component_evidence_and_links():
    assert set(ref.COMPONENT_SOURCES) == set(ref.COMPONENT_IDS.values())
    for source, *locations in ref.COMPONENT_SOURCES.values():
        assert source in ref.SOURCES
        assert set(locations) <= set(ref.SOURCES[source]["equations"])
    doc = (Path(__file__).resolve().parents[1] / "docs" / "math"
           / "imrphenomxas.md").read_text()
    for component in ref.COMPONENT_IDS.values():
        assert component in doc
