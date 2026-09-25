"""Paper-derived XAS intermediate/ringdown phase checks for Torch and JAX."""

from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomxas_phase_branches as ref


def exact(value):
    return sp.Rational(str(value))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("XAS phase reference requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xas import XASCarrierState
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        return ("torch", XASCarrierState,
                lambda x: torch.as_tensor(x, dtype=torch.float64, device=reference_device))
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import XASCarrierState
    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError as exc:
        pytest.fail(f"JAX {platform} unavailable: {exc}")
    return ("jax", XASCarrierState,
            lambda x: jax.device_put(jnp.asarray(x, dtype=jnp.float64), device))


def test_printed_ansatze_primitives_and_sign_distinction():
    rd = ref.ringdown_derivative()
    assert sp.simplify(sp.diff(ref.ringdown_primitive(), ref.F) - rd) == 0
    paper = ref.intermediate_derivative()
    carrier = ref.intermediate_derivative(carrier_sign=True)
    assert sp.simplify(sp.diff(ref.intermediate_primitive(), ref.F) - paper) == 0
    assert sp.simplify(sp.diff(ref.intermediate_primitive(carrier_sign=True), ref.F)
                       - carrier) == 0
    denominator = (ref.F - ref.FRD)**2 + (2 * ref.FDAMP)**2
    assert sp.simplify(carrier - paper
                       - 8 * ref.C0 * ref.APHI / (ref.ETA * denominator)) == 0


def test_collocation_nodes_and_exact_coefficient_recovery():
    rd_nodes = ref.ringdown_nodes(sp.Rational(1, 20), sp.Rational(1, 10),
                                  sp.Rational(1, 50))
    assert rd_nodes[0] == sp.Rational(1, 20)
    assert rd_nodes[3] == sp.Rational(1, 10)
    assert rd_nodes[4] == sp.Rational(1, 8)
    int_nodes = ref.intermediate_nodes(sp.Rational(1, 40), sp.Rational(3, 40))
    assert int_nodes[0] == sp.Rational(1, 40)
    assert int_nodes[2] == sp.Rational(1, 20)
    assert int_nodes[4] == sp.Rational(3, 40)

    nodes = tuple(sp.Rational(1, k**3) for k in (5, 4, 3, 2, 1))
    rd_coeffs = tuple(map(sp.Rational, (2, -3, 1, 4, 5)))
    rd_values = [ref.ringdown_derivative(f=node, eta=1, f_rd=sp.Rational(1, 10),
                                        f_damp=sp.Rational(1, 40),
                                        coefficients=rd_coeffs, a_phi=1)
                 for node in nodes]
    recovered = ref.ringdown_collocation_coefficients(
        nodes, rd_values, sp.Rational(1, 10), sp.Rational(1, 40))
    assert all(sp.simplify(got - want) == 0
               for got, want in zip(recovered, rd_coeffs))

    nodes = tuple(sp.Rational(k, 100) for k in (5, 6, 7, 8, 9))
    int_coeffs = tuple(map(sp.Rational, (3, -2, 1, 4, -5)))
    for carrier_sign in (False, True):
        values = [ref.intermediate_derivative(
            f=node, eta=1, f_rd=sp.Rational(1, 10),
            f_damp=sp.Rational(1, 40), coefficients=int_coeffs, c0=2,
            a_phi=1, carrier_sign=carrier_sign) for node in nodes]
        recovered = ref.intermediate_collocation_coefficients(
            nodes, values, sp.Rational(1, 10), sp.Rational(1, 40), 2,
            carrier_sign=carrier_sign)
        assert all(sp.simplify(got - want) == 0
                   for got, want in zip(recovered, int_coeffs))


@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (30.0, 30.0, 0.0, 0.0),
    (50.0, 10.0, 0.5, -0.3),
    (45.0, 20.0, 0.35, 0.6),
    (90.0, 30.0, -0.4, -0.6),
])
def test_native_collocation_values(backend, m1, m2, chi1, chi2):
    name, carrier, _ = backend
    if name == "torch":
        from torchwave.approximants.imrphenom_xas import XASParameterFits
    else:
        from jaxwave.approximants.imrphenom_xas import XASParameterFits
    state = carrier(m1, m2, chi1=chi1, chi2=chi2)
    eta, s1, s2 = (getattr(state, key) for key in
                   ("eta", "s1_eff", "s2_eff"))

    def fitted(suffix):
        return -float(getattr(XASParameterFits, f"fit_{suffix}")(eta, s1, s2))

    rd, damp, scale = (exact(float(getattr(state, key))) for key in
                       ("f_rd", "f_damp", "eta"))
    rd_coeffs = tuple(exact(float(getattr(state, key))) for key in
                      ("c_rd_val", "c1_val", "c2_val", "c4_val", "c0_val"))
    int_coeffs = tuple(exact(float(getattr(state, key))) for key in
                       ("b0_val", "b1_val", "b2_val", "b3_val", "b4_val"))
    rd_derivative = ref.ringdown_derivative(
        eta=scale, f_rd=rd, f_damp=damp, coefficients=rd_coeffs, a_phi=1)
    int_derivative = ref.intermediate_derivative(
        eta=scale, f_rd=rd, f_damp=damp, coefficients=int_coeffs,
        c0=rd_coeffs[-1], a_phi=1, carrier_sign=True)

    low, high = (float(getattr(state, key)) for key in
                 ("f_trans_phase", "f_join_rd"))
    # The carrier's transition policy has low=f_MECO-dR, high=f_T+dR/2,
    # dR=0.03(f_T-f_MECO). Recover f_T without importing a frontend helper.
    delta_r = 0.03 * (high - low) / 1.045
    f_t = exact(high - 0.5 * delta_r)
    rd_nodes = ref.ringdown_nodes(f_t, rd, damp)
    v4 = fitted("rd_phase_v4")
    d12, d24, d34, d54 = (fitted(f"rd_phase_{key}")
                             for key in ("d12", "d24", "d34", "d54"))
    rd_targets = (v4 + d24 + d12, v4 + d24, v4 + d34, v4, v4 + d54)
    rd_actual = [float((scale * rd_derivative).subs(ref.F, f).evalf(40))
                 for f in rd_nodes]
    np.testing.assert_allclose(rd_actual, rd_targets, rtol=2e-9, atol=2e-5)

    int_nodes = ref.intermediate_nodes(exact(low), exact(high))
    v3 = v4 + fitted("int_phase_v3m_rdv4")
    v2 = 0.75 * (v4 + fitted("int_phase_v2m_rdv4"))
    v2 += 0.25 * fitted("int_phase_v2")
    v4_int = v3 + fitted("int_phase_d43")
    # The fifth fitted value samples the ringdown derivative at f_T, while
    # the fifth collocation node is high=f_T+dR/2 in both frontends.
    int_targets = (v2, v3, v4_int,
                   float((scale * rd_derivative).subs(ref.F, f_t).evalf(40)))
    int_actual = [float((scale * int_derivative).subs(ref.F, f).evalf(40))
                  for f in int_nodes[1:]]
    np.testing.assert_allclose(int_actual, int_targets, rtol=2e-9, atol=2e-5)


@pytest.mark.parametrize("m1,m2,chi1,chi2", [
    (30.0, 30.0, 0.0, 0.0),
    (50.0, 10.0, 0.5, -0.3),
    (45.0, 20.0, 0.35, 0.6),
    (90.0, 30.0, -0.4, -0.6),
])
def test_full_frontend_phase_branches_and_join(backend, m1, m2, chi1, chi2):
    name, carrier, array = backend
    state = carrier(m1, m2, chi1=chi1, chi2=chi2)
    eta, f_rd, damp = (exact(float(getattr(state, key)))
                       for key in ("eta", "f_rd", "f_damp"))
    rd_coeffs = tuple(exact(float(getattr(state, key))) for key in
                      ("c_rd_val", "c1_val", "c2_val", "c4_val", "c0_val"))
    int_coeffs = tuple(exact(float(getattr(state, key))) for key in
                       ("b0_val", "b1_val", "b2_val", "b3_val", "b4_val"))
    rd = ref.ringdown_primitive(eta=eta, f_rd=f_rd, f_damp=damp,
                                coefficients=rd_coeffs, a_phi=1)
    intermediate = ref.intermediate_primitive(
        eta=eta, f_rd=f_rd, f_damp=damp, coefficients=int_coeffs,
        c0=rd_coeffs[-1], a_phi=1, carrier_sign=True)
    join = exact(float(state.f_join_rd))
    slope, offset = ref.join_offsets(join, intermediate, rd)
    np.testing.assert_allclose(float(state.dt_rd), float(slope.evalf(40)),
                               rtol=3e-11, atol=1e-7)
    np.testing.assert_allclose(float(state.dphi_rd), float(offset.evalf(40)),
                               rtol=3e-11, atol=1e-7)
    joined_rd = rd + slope * ref.F + offset
    assert abs(float((joined_rd - intermediate).subs(ref.F, join).evalf(40))) < 1e-7
    assert abs(float(sp.diff(joined_rd - intermediate, ref.F)
                     .subs(ref.F, join).evalf(40))) < 1e-7

    lower = float(state.f_trans_phase)
    upper = float(state.f_join_rd)
    rd_end = float(state.f_rd + state.f_damp)
    assert 0 < lower < upper < rd_end
    for mf, expression in (
        (np.linspace(lower, upper, 5)[1:-1], intermediate),
        (np.linspace(upper, rd_end, 5)[1:], joined_rd),
    ):
        hz = mf / float(state.m_sec)
        phase = state.eval_phase(array(hz))
        actual = phase.detach().cpu().numpy() if name == "torch" else np.asarray(phase)
        actual = actual - float(state.tshift_sec) * hz - 3 * np.pi / 4
        expected = np.asarray([float(expression.subs(ref.F, exact(f)).evalf(40))
                               for f in mf])
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
