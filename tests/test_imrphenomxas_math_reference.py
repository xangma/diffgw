"""Paper quartic collocation and conditional Torch/JAX helper agreement."""

import ast
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
pytest.importorskip("mpmath")

from diffgw_reference import imrphenomxas_intermediate as ref


def exact(value):
    return sp.Rational(str(value))


@pytest.fixture(params=("torch", "jax"))
def backend(request, reference_device):
    if reference_device == "mps":
        pytest.fail("The quartic reference requires FP64, unavailable on MPS")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        from torchwave.approximants.imrphenom_xas import solve_intermediate_amplitude_quartic

        if reference_device == "cuda" and not torch.cuda.is_available():
            pytest.fail("Torch CUDA unavailable")
        return "torch", solve_intermediate_amplitude_quartic, lambda x: torch.as_tensor(
            x, dtype=torch.float64, device=reference_device
        )

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxwave.approximants.imrphenom_xas import solve_intermediate_amplitude_quartic

    platform = "gpu" if reference_device == "cuda" else "cpu"
    try:
        device = jax.devices(platform)[0]
    except RuntimeError as exc:
        pytest.fail(f"JAX {platform} unavailable: {exc}")
    return "jax", solve_intermediate_amplitude_quartic, lambda x: jax.device_put(
        jnp.asarray(x, dtype=jnp.float64), device
    )


def test_paper_quartic_and_physical_slope_constraints():
    """Recover a known quartic from physical A and A' at Table-2 nodes."""
    f1, f3, k = sp.Rational(1, 50), sp.Rational(1, 10), sp.Rational(3, 2)
    original = tuple(map(sp.Rational, (1, 2, -1, 3, 2)))
    amplitude = ref.paper_amplitude(f1=f1, f3=f3, k=k, coefficients=original)
    slope = sp.diff(amplitude, ref.F)
    f2 = (f1 + f3) / 2
    fitted = ref.physical_collocation_coefficients(
        f1, f3, amplitude.subs(ref.F, f1), slope.subs(ref.F, f1),
        amplitude.subs(ref.F, f2), amplitude.subs(ref.F, f3),
        slope.subs(ref.F, f3), k,
    )
    assert all(sp.simplify(got - want) == 0 for got, want in zip(fitted, original))
    reconstructed = ref.paper_amplitude(f1=f1, f3=f3, k=k, coefficients=fitted)
    assert sp.simplify(reconstructed - amplitude) == 0
    assert sp.degree(ref.inverse_polynomial(f1=f1, f3=f3, coefficients=fitted), ref.F) == 4


@pytest.mark.parametrize("f1,f3,values,slopes", [
    (0.014, 0.08, (1.25, 1.4, 0.92), (1.2, -0.7)),
    (0.008, 0.14, (0.7, 1.02, 1.3), (-0.5, 2.0)),
    (0.025, 0.035, (1.8, 1.72, 1.64), (0.3, -0.4)),
])
def test_frontend_quartic_against_scalar_matrix_solve(
    backend, f1, f3, values, slopes
):
    """Check coefficients and the amplitude/slope at all five constraints."""
    _, solve, array = backend
    v1, v2, v3 = values
    d1, d3 = slopes
    expected = ref.collocation_coefficients(*map(exact, (f1, f3, v1, d1, v2, v3, d3)))
    actual = solve(*map(array, (f1, f3, v1, d1, v2, v3, d3)))
    np.testing.assert_allclose([float(x) for x in actual], [float(x) for x in expected],
                               rtol=2e-13, atol=2e-13)
    amp = ref.normalized_amplitude(f1=exact(f1), f3=exact(f3), coefficients=expected)
    slope = sp.diff(amp, ref.F)
    for f, v in ((f1, v1), ((f1 + f3) / 2, v2), (f3, v3)):
        np.testing.assert_allclose(float(sp.N(amp.subs(ref.F, exact(f)), 40)), v,
                                   rtol=2e-13, atol=2e-13)
    for f, d in ((f1, d1), (f3, d3)):
        np.testing.assert_allclose(float(sp.N(slope.subs(ref.F, exact(f)), 40)), d,
                                   rtol=2e-12, atol=2e-12)


def test_midpoint_coefficient_derivative(backend):
    """Differentiate the independent matrix solve and each native helper."""
    name, solve, array = backend
    v2 = sp.Symbol("v2", positive=True)
    coefficient = ref.collocation_coefficients(
        exact(.014), exact(.08), exact(1.25), exact(1.2), v2,
        exact(.92), exact(-.7)
    )[2]
    expected = float(sp.N(sp.diff(coefficient, v2).subs(v2, exact(1.4)), 40))
    arguments = tuple(map(array, (.014, .08, 1.25, 1.2, 1.4, .92, -.7)))
    if name == "torch":
        import torch

        variable = arguments[4].clone().requires_grad_()
        actual = torch.autograd.grad(solve(*arguments[:4], variable, *arguments[5:])[2],
                                     variable)[0]
    else:
        import jax

        actual = jax.grad(lambda value: solve(*arguments[:4], value, *arguments[5:])[2])(
            arguments[4]
        )
    np.testing.assert_allclose(float(actual), expected, rtol=2e-12, atol=2e-12)


def test_batched_quartic_and_jax_jit(backend):
    name, solve, array = backend
    cases = (
        (.014, .08, 1.25, 1.2, 1.4, .92, -.7),
        (.008, .14, .7, -.5, 1.02, 1.3, 2.0),
    )
    columns = tuple(array([case[index] for case in cases]) for index in range(7))
    if name == "jax":
        import jax

        solve = jax.jit(solve)
    actual = np.asarray([np.asarray(value) for value in solve(*columns)]).T
    expected = np.asarray([
        [float(value) for value in ref.collocation_coefficients(*map(exact, case))]
        for case in cases
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_component_evidence_and_implementation_links():
    assert set(ref.COMPONENT_SOURCES) == set(ref.COMPONENT_IDS.values())
    for source, *locations in ref.COMPONENT_SOURCES.values():
        assert source in ref.SOURCES
        assert set(locations) <= set(ref.SOURCES[source]["equations"])
    root = Path(__file__).resolve().parents[1]
    for frontend in ("torchwave", "jaxwave"):
        path = root / "src" / frontend / "approximants" / "imrphenom_xas.py"
        tree = ast.parse(path.read_text())
        assert any(isinstance(node, ast.FunctionDef)
                   and node.name == "solve_intermediate_amplitude_quartic"
                   for node in tree.body)
