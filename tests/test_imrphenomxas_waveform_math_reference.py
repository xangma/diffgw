"""Absolute XAS complex-strain checks against the composed scalar reference."""

from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomxas_waveform as ref


PARAMETERS = (
    "f_rd", "f_damp", "width_rd", "lam", "a_r", "f1_col", "f4_col",
    "f_trans_amp", "f_peak", "f_trans_phase", "f_join_rd", "tshift_sec",
    "dt_rd", "dphi_rd", "c_rd_val", "c1_val", "c2_val", "c4_val", "c0_val",
    *(f"rho{i}" for i in range(1, 4)),
    *(f"c{i}_amp" for i in range(5)),
    *(f"sig{i}" for i in range(5)),
    *(f"b{i}_val" for i in range(5)),
)
CASES = ((30., 30., 0., 0.), (50., 10., .5, -.3))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("XAS composed reference requires FP64")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xas import XASCarrierState
        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        return "torch", XASCarrierState, lambda x: torch.as_tensor(
            x, dtype=torch.float64, device=reference_device)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import XASCarrierState
    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError as exc:
        pytest.fail(f"JAX {platform} unavailable: {exc}")
    return "jax", XASCarrierState, lambda x: jax.device_put(
        jnp.asarray(x, dtype=jnp.float64), device)


def as_numpy(value, name):
    return np.asarray(value.detach().cpu() if name == "torch" else value)


def snapshot(state):
    """Boundary adapter: calibrated outputs are data, never imported by ref."""
    return {name: str(float(getattr(state, name))) for name in PARAMETERS}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reference_region", (None, "inspiral", "intermediate", "ringdown"))
def test_full_complex_polarizations(backend, case, reference_region):
    name, carrier, array = backend
    probe = carrier(*case[:2], chi1=case[2], chi2=case[3])
    m_sec = float(probe.m_sec)
    p0, p1 = float(probe.f_trans_phase), float(probe.f_join_rd)
    f_ref = {None: None, "inspiral": .6*p0/m_sec,
             "intermediate": (p0+p1)/(2*m_sec),
             "ringdown": 1.1*p1/m_sec}[reference_region]
    state = carrier(*case[:2], chi1=case[2], chi2=case[3], f_ref=f_ref)
    spec = ref.compose(*case, 470., snapshot(state))
    a0, a1 = map(float, spec.amp_joins)
    p0, p1 = map(float, spec.phase_joins)
    mf = np.array(sorted({
        .008, .8*a0, .999*a0, 1.001*a0,
        .999*a1, 1.001*a1, .999*p0, 1.001*p0,
        .999*p1, 1.001*p1, float(state.f_rd),
        1.2*float(state.f_rd), .2,
    }))
    frequencies = mf / m_sec
    hp, hc, phase = state.eval_waveform(
        array(frequencies), distance=array(470.), inclination=array(.7),
        phi_c=array(.23), t_c=array(.003), return_phase=True)
    expected = [spec.evaluate(f, inclination=.7, phic=.23, tc=.003,
                              f_ref_hz=f_ref) for f in frequencies]
    expected_hp = np.array([row[0] for row in expected])
    expected_hc = np.array([row[1] for row in expected])
    expected_phase = np.array([row[2] for row in expected])
    np.testing.assert_allclose(as_numpy(hp, name), expected_hp, rtol=2e-8, atol=1e-31)
    np.testing.assert_allclose(as_numpy(hc, name), expected_hc, rtol=2e-8, atol=1e-31)
    np.testing.assert_allclose(as_numpy(phase, name), expected_phase,
                               rtol=2e-9, atol=3e-6)
    masked = state.eval_waveform(array([-10., 0.]), distance=array(470.))
    assert np.all(as_numpy(masked[0], name) == 0)
    assert np.all(as_numpy(masked[1], name) == 0)
    assert spec.evaluate(0.)[:2] == (0j, 0j)


def test_polarization_and_extrinsic_algebra():
    a, psi, f, d, t = sp.symbols("a psi f d t", positive=True)
    hp, hc = ref.polarizations(a, psi, 0)
    assert sp.simplify(hc + sp.I*hp) == 0
    hp, hc = ref.polarizations(a, psi, sp.pi/2)
    assert sp.simplify(hc) == 0
    assert sp.simplify(hp - a*sp.exp(-sp.I*psi)/2) == 0
    _, dhc = ref.extrinsic_derivatives(hp, hc, f, d, "inclination",
                                       inclination=sp.pi/2)
    assert sp.simplify(dhc-sp.I*a*sp.exp(-sp.I*psi)) == 0
    for parameter, multiplier in (("distance_mpc", -1/d),
                                  ("tc", -2*sp.pi*sp.I*f),
                                  ("phic", 2*sp.I)):
        dhp, dhc = ref.extrinsic_derivatives(hp, hc, f, d, parameter)
        assert sp.simplify(dhp-multiplier*hp) == 0
        assert sp.simplify(dhc-multiplier*hc) == 0
    assert sp.simplify(ref.physical_scale(f, t, 2*d, sp.Rational(1, 4))
                       / ref.physical_scale(f, t, d, sp.Rational(1, 4))
                       - sp.Rational(1, 2)) == 0


@pytest.mark.parametrize("mf_branch", ("inspiral", "ringdown"))
@pytest.mark.parametrize("parameter", ("distance_mpc", "inclination", "tc", "phic"))
def test_native_extrinsic_gradients(backend, mf_branch, parameter):
    """Differentiate full complex strain, holding calibrated intrinsic data fixed."""
    name, carrier, array = backend
    case = CASES[1]
    state = carrier(*case[:2], chi1=case[2], chi2=case[3])
    spec = ref.compose(*case, 470., snapshot(state))
    mf = (.6*float(state.f_trans_amp) if mf_branch == "inspiral"
          else 1.2*float(state.f_peak))
    frequency = mf / float(state.m_sec)
    values = {"distance_mpc": 470., "inclination": .7, "tc": .003,
              "phic": .23}
    hp, hc, _ = spec.expressions(
        frequency, inclination=.7, tc=.003, phic=.23)
    expected = ref.extrinsic_derivatives(
        hp, hc, ref.exact(frequency), ref.exact(470.), parameter,
        inclination=ref.exact(.7))
    expected = np.array([complex(sp.N(x, 45)) for x in expected])
    reference = np.array([complex(sp.N(x, 45)) for x in (hp, hc)])

    def output(x):
        args = {key: array(x if key == parameter else value)
                for key, value in values.items()}
        plus, cross = state.eval_waveform(
            array([frequency]), distance=args["distance_mpc"],
            inclination=args["inclination"], t_c=args["tc"],
            phi_c=args["phic"])
        return plus[0], cross[0]

    if name == "torch":
        import torch
        x = array(values[parameter]).detach().clone().requires_grad_(True)
        plus, cross = output(x)
        actual = np.array([
            complex(float(torch.autograd.grad(plus.real, x, retain_graph=True)[0]),
                    float(torch.autograd.grad(plus.imag, x, retain_graph=True)[0])),
            complex(float(torch.autograd.grad(cross.real, x, retain_graph=True)[0]),
                    float(torch.autograd.grad(cross.imag, x)[0])),
        ])
    else:
        import jax
        import jax.numpy as jnp
        def parts(x):
            plus, cross = output(x)
            return jnp.stack([plus.real, plus.imag, cross.real, cross.imag])
        vector = np.asarray(jax.jacfwd(parts)(array(values[parameter])))
        actual = np.array([complex(*vector[:2]), complex(*vector[2:])])
    np.testing.assert_allclose(actual/reference, expected/reference,
                               rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("mf_branch", ("inspiral", "intermediate", "ringdown"))
def test_native_complex_frequency_gradient(backend, mf_branch):
    name, carrier, array = backend
    case = CASES[1]
    probe = carrier(*case[:2], chi1=case[2], chi2=case[3])
    f_ref = 1.1*float(probe.f_trans_phase)/float(probe.m_sec)
    state = carrier(*case[:2], chi1=case[2], chi2=case[3], f_ref=f_ref)
    spec = ref.compose(*case, 470., snapshot(state))
    a0, a1 = map(float, spec.amp_joins)
    mf = {"inspiral": .6*a0, "intermediate": (a0+a1)/2,
          "ringdown": 1.2*a1}[mf_branch]
    frequency = mf/float(state.m_sec)
    expected = np.array([complex(sp.N(x, 45)) for x in
                         spec.frequency_derivatives(
                             frequency, inclination=.7, tc=.003, phic=.23,
                             f_ref_hz=f_ref)])
    reference = np.array([complex(sp.N(x, 45)) for x in
                          spec.expressions(frequency, inclination=.7, tc=.003,
                                           phic=.23, f_ref_hz=f_ref)[:2]])

    def output(f):
        hp, hc = state.eval_waveform(
            f, distance=array(470.), inclination=array(.7),
            t_c=array(.003), phi_c=array(.23))
        return hp[0], hc[0]

    if name == "torch":
        import torch
        f = array([frequency]).detach().clone().requires_grad_(True)
        hp, hc = output(f)
        actual = np.array([
            complex(float(torch.autograd.grad(hp.real, f, retain_graph=True)[0][0]),
                    float(torch.autograd.grad(hp.imag, f, retain_graph=True)[0][0])),
            complex(float(torch.autograd.grad(hc.real, f, retain_graph=True)[0][0]),
                    float(torch.autograd.grad(hc.imag, f)[0][0])),
        ])
    else:
        import jax
        import jax.numpy as jnp
        def parts(f):
            hp, hc = output(jnp.asarray([f]))
            return jnp.stack([hp.real, hp.imag, hc.real, hc.imag])
        vector = np.asarray(jax.jacfwd(parts)(array(frequency)))
        actual = np.array([complex(*vector[:2]), complex(*vector[2:])])
    np.testing.assert_allclose(actual/reference, expected/reference,
                               rtol=2e-7, atol=2e-7)


def test_component_evidence_and_links():
    assert set(ref.COMPONENT_IDS.values()) == set(ref.COMPONENT_SOURCES)
    assert ref.COMPONENT_VERSIONS[:4] == (
        ref.amp.SPEC_VERSION, ref.middle.SPEC_VERSION,
        ref.inspiral.SPEC_VERSION, ref.phase.SPEC_VERSION)
    doc = (Path(__file__).resolve().parents[1] / "docs" / "math"
           / "imrphenomxas.md").read_text()
    for component in ref.COMPONENT_IDS.values():
        assert component in doc
