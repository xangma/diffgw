"""Independent high-precision checks for published XPNR component algebra.

No inherited fit tensor, LAL source, or backend output supplies an expectation.
The public XPNR composition remains provisional; this gate covers only the
published expressions with caller-supplied boundary/calibration inputs.
"""

import cmath
import math

import pytest

pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import xpnr_angles as scalar
from diffgw_reference import xpnr_symbolic as symbolic


INPUTS = {
    "f": .08, "f0": .06, "f1": .1, "fc": .1,
    "y0": .2, "y1": .3, "dy0": .4, "dy1": -.2,
    "y": .27, "y_c": .3, "dy_c": .4, "target": .35, "dtarget": -.1,
    "eta": .24, "chi": .4, "c": .6, "q": 1.5, "theta_f": .45,
    "lam_a1": .2, "lam_a2": .3, "lam_a3": .4, "lam_a4": .5,
    "lam_b0": .2, "lam_b1": .1, "lam_b2": .2, "lam_b3": 5.4,
    "lam_b4": 15., "lam_b5": .2, "lam_bf": .1,
    "a1": .01, "a2": -.03, "a3": .04, "a4": .05,
    "b0": .4, "b1": .2, "b2": -.1, "b3": .3, "b4": 200., "b5": -.04,
    "m1": 30., "m2": 20., "sx1": .2, "sy1": .1, "sz1": .3,
    "sx2": -.1, "sy2": .05, "sz2": -.2, "lx": 0., "ly": 0., "lz": 1.,
    "cutoff_feature": .08, "fmax": .1, "finf": .12, "fmin": .08,
    "ddmax": -4., "dddmax": -100., "dinf": .5,
    "beta_rd": .12, "kappa": 2.,
    "v": .25, "iota": .3, "delta": .2,
    "chi_perp_input": .3, "chi_parallel_input": .1,
    "mode": 3, "rd22": .08, "rdlm": .09,
    "lam_k": .7, "u_k": -.2,
    "omega_lm": .5, "omega_22": .6, "omega_21": .4, "cos_beta_rd": .8,
    "a_s": .8, "a_a": .1, "phi_s": .5, "phi_a": -.3,
}


def _expected():
    p = INPUTS
    mapped = scalar.single_spin_map_from_transition_vectors(
        p["m1"], p["m2"], (p["sx1"], p["sy1"], p["sz1"]),
        (p["sx2"], p["sy2"], p["sz2"]), (p["lx"], p["ly"], p["lz"]))
    frozen = scalar.freeze_fit_coordinates(p["eta"], p["chi"], p["c"], p["q"])
    alpha = scalar.alpha_parameters(*[p[k] for k in (
        "chi", "c", "lam_a1", "lam_a2", "lam_a3", "lam_a4")])
    beta = scalar.beta_parameters(*[p[k] for k in (
        "theta_f", "chi", "c", "lam_b0", "lam_b1", "lam_b2", "lam_b3",
        "lam_b4", "lam_b5")])
    attach = (p["f"], p["f0"], p["f1"], p["y0"], p["dy0"], p["y1"], p["dy1"])
    ext = (p["f"], p["f0"], p["f1"], p["y0"], p["y1"], p["dy1"])
    result = {
        "map.chi_effective": mapped.chi_effective,
        "map.chi_parallel": mapped.chi_parallel,
        "map.chi_perpendicular": mapped.chi_perpendicular,
        "map.chi": mapped.chi,
        "map.cos_theta": mapped.cos_theta_ls,
        "fit.eta_frozen": frozen.eta, "fit.chi_frozen": frozen.chi,
        "fit.beta_rd": scalar.beta_ringdown(p["theta_f"], p["chi"], p["c"], p["lam_bf"]),
        "angles.alpha_mr": scalar.alpha_merger_ringdown(*[p[k] for k in
            ("f", "a1", "a2", "a3", "a4")]),
        "angles.beta_mr": scalar.beta_merger_ringdown(*[p[k] for k in
            ("f", "b0", "b1", "b2", "b3", "b4", "b5")]),
        "attach.alpha": scalar.alpha_attachment(*attach),
        "attach.beta": scalar.beta_connection(*attach),
        "attach.beta_cutoff": scalar.beta_feature_cutoff_frequency(p["cutoff_feature"]),
        "attach.beta_turnover": scalar.beta_turnover_connection_frequency(
            p["fmax"],p["ddmax"],p["dddmax"],p["dinf"]),
        "attach.beta_lower": scalar.beta_lower_connection_frequency(
            p["fmax"],p["finf"],p["fmin"],p["ddmax"],p["dddmax"],p["dinf"]),
        "attach.taper": scalar.tapered_opening_angle(p["f"],p["fc"],p["y0"],p["y1"]),
        "attach.beta_bound": scalar.bounded_opening_angle(p["y"]),
        "attach.emission_beta": scalar.optimal_emission_direction_beta(
            p["v"],p["iota"],p["eta"],p["delta"],p["chi_perp_input"],p["chi_parallel_input"]),
        "attach.alpha_exterior": scalar.alpha_exterior_continuation(*ext),
        "attach.beta_exterior": scalar.beta_exterior_continuation(
            *ext,p["beta_rd"],p["kappa"]),
        "attach.beta_rescaled": scalar.beta_final_rescaling(
            p["f"],p["fc"],p["y"],p["y_c"],p["dy_c"],p["target"],p["dtarget"]),
        "carrier.deformation": scalar.deformed_carrier_coefficient(
            p["lam_k"],p["chi"],p["c"],p["u_k"]),
        "carrier.effective_ringdown": scalar.effective_ringdown_angular_frequency(
            p["omega_lm"],p["mode"],p["cos_beta_rd"],p["omega_22"],p["omega_21"]),
        "carrier.asymmetric_plus": p["a_s"]*cmath.exp(1j*p["phi_s"])+p["a_a"]*cmath.exp(1j*p["phi_a"]),
        "carrier.asymmetric_minus": p["a_s"]*cmath.exp(-1j*p["phi_s"])-p["a_a"]*cmath.exp(-1j*p["phi_a"]),
        "higher_mode.frequency_map": scalar.higher_mode_frequency_map(
            p["f"],p["mode"],p["f0"],p["f1"],p["rd22"],p["rdlm"]),
        "window.endpoint_complete": scalar.angle_calibration_window(p["q"],p["chi"]),
    }
    result.update(zip((f"fit.a{i}" for i in range(1,5)),alpha))
    result.update(zip((f"fit.b{i}" for i in range(6)),beta))
    return result


@pytest.mark.parametrize("name", sorted(symbolic.expressions()))
def test_symbolic_components_against_independent_scalar_forms(name):
    expected = _expected()
    assert set(expected) == set(symbolic.expressions())
    actual = complex(symbolic.evaluate(name, INPUTS, dps=70))
    assert actual == pytest.approx(expected[name], rel=2e-12, abs=2e-12)


def test_polynomial_preserves_decimal_precision_and_rejects_unpublished_exponents():
    import sympy as sp

    polynomial = symbolic.calibration_polynomial({(0,0,0): "0.10000000000000001",
                                                  (2,1,3): "-0.00037"})
    assert polynomial.subs({sp.Symbol("eta",real=True): sp.Rational(1,4),
                            sp.Symbol("chi",real=True): sp.Rational(1,2),
                            sp.Symbol("c",real=True): sp.Rational(3,5)}) == (
                                sp.Rational("0.10000000000000001")-
                                sp.Rational("0.00037")*sp.Rational(1,4)**2*
                                sp.Rational(1,2)*sp.Rational(3,5)**3)
    with pytest.raises(ValueError, match="outside"):
        symbolic.calibration_polynomial({(4,0,0): "1"})


@pytest.mark.parametrize("name", ["attach.alpha", "attach.beta",
                                    "attach.alpha_exterior", "attach.beta_exterior"])
def test_symbolic_join_values_and_endpoint_slopes(name):
    p = dict(INPUTS)
    for f, value, slope in ((p["f0"],p["y0"],p["dy0"]),
                            (p["f1"],p["y1"],p["dy1"])):
        if "exterior" in name and f == p["f0"]:
            check_slope = False
        else:
            check_slope = True
        p["f"] = f
        assert float(symbolic.evaluate(name,p)) == pytest.approx(value,abs=1e-12)
        if check_slope:
            assert float(symbolic.evaluate(name,p,derivative="f")) == pytest.approx(slope,abs=1e-10)


def test_high_precision_derivative_stabilizes_and_is_independent_of_scalar_float():
    p = dict(INPUTS)
    p["f"] = .099999
    low = symbolic.evaluate("attach.beta_exterior",p,derivative="f",dps=40)
    high = symbolic.evaluate("attach.beta_exterior",p,derivative="f",dps=80)
    assert abs(low-high) < 1e-35
    assert math.isfinite(float(high))


NATIVE_COMPONENTS = (
    ("attach.alpha", "alpha_attachment", ("f","f0","f1","y0","dy0","y1","dy1"), "f"),
    ("attach.beta", "beta_connection", ("f","f0","f1","y0","dy0","y1","dy1"), "f"),
    ("attach.beta_cutoff", "beta_feature_cutoff_frequency", ("cutoff_feature",), "cutoff_feature"),
    ("attach.beta_turnover", "beta_turnover_connection_frequency",
     ("fmax","ddmax","dddmax","dinf"), "dinf"),
    ("attach.beta_lower", "beta_lower_connection_frequency",
     ("fmax","finf","fmin","ddmax","dddmax","dinf"), "dinf"),
    ("attach.taper", "tapered_opening_angle", ("f","fc","y0","y1"), "f"),
    ("attach.beta_bound", "bounded_opening_angle", ("y",), "y"),
    ("attach.emission_beta", "optimal_emission_direction_beta",
     ("v","iota","eta","delta","chi_perp_input","chi_parallel_input"), "v"),
    ("attach.alpha_exterior", "alpha_exterior_continuation",
     ("f","f0","f1","y0","y1","dy1"), "f"),
    ("attach.beta_exterior", "beta_exterior_continuation",
     ("f","f0","f1","y0","y1","dy1","beta_rd","kappa"), "f"),
    ("attach.beta_rescaled", "beta_final_rescaling",
     ("f","fc","y","y_c","dy_c","target","dtarget"), "f"),
)


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize("component,function,arguments,differentiated", NATIVE_COMPONENTS)
def test_native_attachment_value_and_gradient_against_symbolic_reference(
        backend,component,function,arguments,differentiated):
    p = INPUTS
    expected = float(symbolic.evaluate(component,p,dps=70))
    expected_gradient = float(symbolic.evaluate(component,p,derivative=differentiated,dps=70))
    index = arguments.index(differentiated)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants import pnr_attachment
        args = [torch.tensor(p[key],dtype=torch.float64,requires_grad=(i == index))
                for i,key in enumerate(arguments)]
        result = getattr(pnr_attachment,function)(*args)
        gradient, = torch.autograd.grad(result,args[index])
        actual, actual_gradient = result.item(),gradient.item()
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants import pnr_attachment
        args = tuple(jnp.asarray(p[key],dtype=jnp.float64) for key in arguments)
        fn = getattr(pnr_attachment,function)
        actual = float(jax.jit(fn)(*args))
        actual_gradient = float(jax.jit(jax.grad(fn,argnums=index))(*args))
    assert actual == pytest.approx(expected,rel=2e-11,abs=2e-11)
    assert actual_gradient == pytest.approx(expected_gradient,rel=2e-9,abs=2e-9)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_batch_and_supported_jax_transforms_against_symbolic_join(backend):
    points = (.07,.08,.09)
    expected = [float(symbolic.evaluate("attach.beta",dict(INPUTS,f=f),dps=60))
                for f in points]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.pnr_attachment import beta_connection
        grid = torch.tensor(points,dtype=torch.float64,requires_grad=True)
        p = INPUTS
        result = beta_connection(grid,p["f0"],p["f1"],p["y0"],p["dy0"],p["y1"],p["dy1"])
        gradient, = torch.autograd.grad(result.sum(),grid)
        actual = result.detach().tolist()
        derivatives = gradient.tolist()
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.pnr_attachment import beta_connection
        p = INPUTS
        def single(f):
            return beta_connection(f,p["f0"],p["f1"],p["y0"],p["dy0"],p["y1"],p["dy1"])
        grid = jnp.asarray(points,dtype=jnp.float64)
        actual = list(map(float,jax.jit(jax.vmap(single))(grid)))
        derivatives = list(map(float,jax.jit(jax.vmap(jax.grad(single)))(grid)))
    assert actual == pytest.approx(expected,rel=2e-12,abs=2e-12)
    assert derivatives == pytest.approx([
        float(symbolic.evaluate("attach.beta",dict(INPUTS,f=f),derivative="f",dps=60))
        for f in points],rel=2e-10,abs=2e-10)


@pytest.mark.parametrize("value", (0.0, .3, math.pi/2, 2.8, math.pi, 12.0))
def test_opening_angle_bound_in_published_domain_and_outside(value):
    result = scalar.bounded_opening_angle(value)
    symbolic_result = float(symbolic.evaluate("attach.beta_bound",dict(INPUTS,y=value),dps=90))
    assert result == pytest.approx(symbolic_result,abs=3e-14)
    assert 0 <= result <= math.pi


def test_opening_angle_bound_has_unit_central_slope():
    actual = symbolic.evaluate("attach.beta_bound",dict(INPUTS,y=math.pi/2),
                               derivative="y",dps=80)
    assert float(actual) == pytest.approx(1.0,abs=1e-12)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_native_opening_bound_batch_and_central_gradient(backend):
    points = (0.0, .3, math.pi/2, 2.8, math.pi, 12.0)
    expected = [float(symbolic.evaluate("attach.beta_bound",dict(INPUTS,y=y),dps=90))
                for y in points]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.pnr_attachment import bounded_opening_angle
        values = torch.tensor(points,dtype=torch.float64,requires_grad=True)
        result = bounded_opening_angle(values)
        gradient, = torch.autograd.grad(result[2],values)
        actual = result.detach().tolist()
        central_derivative = gradient[2].item()
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from jaxwave.approximants.pnr_attachment import bounded_opening_angle
        values = jnp.asarray(points,dtype=jnp.float64)
        actual = list(map(float,jax.jit(jax.vmap(bounded_opening_angle))(values)))
        central_derivative = float(jax.jit(jax.grad(bounded_opening_angle))(
            jnp.asarray(math.pi/2,dtype=jnp.float64)))
    assert actual == pytest.approx(expected,abs=3e-14)
    assert central_derivative == pytest.approx(1.0,abs=1e-12)
