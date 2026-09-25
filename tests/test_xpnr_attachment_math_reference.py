"""Paper-form checks for the separately scoped XPNR angle joins.

Hamilton et al., arXiv:2507.02604v1, Eqs. (17)--(21), Hamilton et al.,
arXiv:2107.08876v1, Eqs. (41)--(45), and the correction stated in Thompson
et al., arXiv:2312.10025v1. These tests use
synthetic coefficients in the printed ansatzes, not inherited fit tensors or
LAL source. The public waveform is out of scope.
"""

import math

import pytest

from diffgw_reference import xpnr_angles as scalar


START, END = 0.04, 0.08
KAPPA, BETA_RD = 15.0, 0.3
EMISSION_ARGS = (
    0.27, 0.8, 0.22, 0.35,
    0.67 * math.sqrt(1 - 0.28**2), -0.67 * 0.28,
)


def _alpha_attach(f):
    return 2.3 * f**2 - 1.4 * f + 0.7 - 0.02 / f


def _alpha_attach_prime(f):
    return 4.6 * f - 1.4 + 0.02 / f**2


def _beta_connect(f):
    return 0.4 + 2.1 * f - 8.0 * f**2 + 11.0 * f**3


def _beta_connect_prime(f):
    return 2.1 - 16.0 * f + 33.0 * f**2


def _alpha_exterior(f):
    return -(0.1 + 0.0005 / f**2 - 1.0e-8 / f**4)


def _alpha_exterior_prime(f):
    return 0.001 / f**3 - 4.0e-8 / f**5


def _beta_exterior(f):
    return BETA_RD + math.exp(-KAPPA * f) * (
        0.0002 / f**2 + 0.00002 / f**3 + 0.000001 / f**4
    )


def _beta_exterior_prime(f):
    return -KAPPA * (_beta_exterior(f) - BETA_RD) + math.exp(-KAPPA * f) * (
        -0.0004 / f**3 - 0.00006 / f**4 - 0.000004 / f**5
    )


CASES = {
    "alpha_attachment": (_alpha_attach, _alpha_attach_prime),
    "beta_connection": (_beta_connect, _beta_connect_prime),
    "alpha_exterior_continuation": (_alpha_exterior, _alpha_exterior_prime),
    "beta_exterior_continuation": (_beta_exterior, _beta_exterior_prime),
}


def _arguments(name, f):
    source, derivative = CASES[name]
    if name in ("alpha_attachment", "beta_connection"):
        return f, START, END, source(START), derivative(START), source(END), derivative(END)
    args = (f, START, END, source(START), source(END), derivative(END))
    return args + (BETA_RD, KAPPA) if name == "beta_exterior_continuation" else args


def _points(name):
    points = (START, 0.05, 0.065, END)
    return points + (0.1,) if "continuation" in name else points


@pytest.mark.parametrize("name", CASES)
def test_xpnr_attachment_scalar_matches_printed_ansatz(name):
    source, _ = CASES[name]
    kernel = getattr(scalar, name)
    for f in _points(name):
        assert math.isclose(kernel(*_arguments(name, f)), source(f), rel_tol=2e-12, abs_tol=2e-12)


@pytest.mark.parametrize("name", CASES)
def test_xpnr_attachment_torch_value_and_join_derivative(name):
    import torch

    from torchwave.approximants import pnr_attachment

    source, derivative = CASES[name]
    kernel = getattr(pnr_attachment, name)
    for f in _points(name):
        f_tensor = torch.tensor(f, dtype=torch.float64, requires_grad=True)
        args = tuple(torch.as_tensor(arg, dtype=torch.float64) for arg in _arguments(name, f)[1:])
        output = kernel(f_tensor, *args)
        slope = torch.autograd.grad(output, f_tensor)[0]
        assert math.isclose(output.item(), source(f), rel_tol=2e-12, abs_tol=2e-12)
        assert math.isclose(slope.item(), derivative(f), rel_tol=2e-10, abs_tol=2e-10)


@pytest.mark.parametrize("name", CASES)
def test_xpnr_attachment_jax_jit_value_and_join_derivative(name):
    import jax
    import jax.numpy as jnp

    from jaxwave.approximants import pnr_attachment

    source, derivative = CASES[name]
    kernel = getattr(pnr_attachment, name)
    args = tuple(jnp.asarray(arg, dtype=jnp.float64) for arg in _arguments(name, START)[1:])
    compiled = jax.jit(lambda f: kernel(f, *args))
    compiled_slope = jax.jit(jax.grad(lambda f: kernel(f, *args)))
    for f in _points(name):
        f_array = jnp.asarray(f, dtype=jnp.float64)
        assert math.isclose(float(compiled(f_array)), source(f), rel_tol=2e-12, abs_tol=2e-12)
        assert math.isclose(float(compiled_slope(f_array)), derivative(f), rel_tol=2e-10, abs_tol=2e-10)


def _base_beta(f):
    return 0.4 + 2.0 * f + 0.5 * f**2


def _base_beta_prime(f):
    return 2.0 + f


def _rescaled_beta(f):
    return _base_beta(f) * (1.0 - 0.7 * f + 2.4 * f**2)


def _rescaled_beta_prime(f):
    return _base_beta_prime(f) * (1.0 - 0.7 * f + 2.4 * f**2) + _base_beta(f) * (-0.7 + 4.8 * f)


def _rescale_boundary():
    return END, _base_beta(END), _base_beta_prime(END), _rescaled_beta(END), _rescaled_beta_prime(END)


def test_xpnr_beta_final_rescaling_scalar_matches_printed_quadratic():
    for f in (0.0, 0.02, 0.05, END):
        actual = scalar.beta_final_rescaling(f, END, _base_beta(f), *_rescale_boundary()[1:])
        assert math.isclose(actual, _rescaled_beta(f), rel_tol=2e-12, abs_tol=2e-12)
    with pytest.raises(ValueError, match="nonzero base beta"):
        scalar.beta_final_rescaling(0.05, END, _base_beta(0.05), 0.0, 1.0, 0.5, 1.0)


def test_xpnr_beta_final_rescaling_torch_value_and_join_derivative():
    import torch

    from torchwave.approximants.pnr_attachment import beta_final_rescaling

    boundary = tuple(torch.tensor(value, dtype=torch.float64) for value in _rescale_boundary())
    for f in (0.0, 0.02, 0.05, END):
        mf = torch.tensor(f, dtype=torch.float64, requires_grad=True)
        actual = beta_final_rescaling(mf, boundary[0], _base_beta(mf), *boundary[1:])
        slope = torch.autograd.grad(actual, mf)[0]
        assert math.isclose(actual.item(), _rescaled_beta(f), rel_tol=2e-12, abs_tol=2e-12)
        assert math.isclose(slope.item(), _rescaled_beta_prime(f), rel_tol=2e-10, abs_tol=2e-10)


def test_xpnr_beta_final_rescaling_jax_jit_value_and_join_derivative():
    import jax
    import jax.numpy as jnp

    from jaxwave.approximants.pnr_attachment import beta_final_rescaling

    boundary = tuple(jnp.asarray(value, dtype=jnp.float64) for value in _rescale_boundary())
    compiled = jax.jit(lambda f: beta_final_rescaling(f, boundary[0], _base_beta(f), *boundary[1:]))
    compiled_slope = jax.jit(jax.grad(lambda f: beta_final_rescaling(f, boundary[0], _base_beta(f), *boundary[1:])))
    for f in (0.0, 0.02, 0.05, END):
        mf = jnp.asarray(f, dtype=jnp.float64)
        assert math.isclose(float(compiled(mf)), _rescaled_beta(f), rel_tol=2e-12, abs_tol=2e-12)
        assert math.isclose(float(compiled_slope(mf)), _rescaled_beta_prime(f), rel_tol=2e-10, abs_tol=2e-10)


def _tapered_opening_angle(f, connection, two_spin, single_spin):
    if f > connection:
        return single_spin
    return single_spin + math.cos(math.pi * f / (2 * connection)) ** 2 * (two_spin - single_spin)


def _optimal_emission_direction_beta(v, iota, eta, delta, chi_perpendicular, chi_parallel):
    sin_iota, cos_iota = math.sin(iota), math.cos(iota)
    sin_half_iota, cos_half_iota = math.sin(iota / 2), math.cos(iota / 2)
    mass_spin = (6 + 6 * delta + 5 * eta) * chi_perpendicular
    aligned_spin = (1 + delta - eta) * chi_parallel
    c0 = 84 * sin_iota
    c2 = (110 * eta - 214) * sin_iota
    c3 = -7 * mass_spin * (2 * cos_iota - 1) + 56 * (3 * math.pi - aligned_spin) * sin_iota
    d0 = 84 * cos_half_iota
    d2 = (110 * eta - 214) * cos_half_iota
    d3 = 14 * mass_spin * sin_half_iota + 56 * cos_half_iota * (3 * math.pi - aligned_spin)
    return 2 * math.atan(
        ((c0 + c2 * v**2 + c3 * v**3) / cos_half_iota)
        / (2 * (d0 + d2 * v**2 + d3 * v**3))
    )


def test_beta_taper_scalar_matches_published_piecewise_equation():
    connection, two_spin, single_spin = 0.08, 1.1, 0.7
    for f in (0.0, 0.03, connection, 0.1):
        assert math.isclose(
            scalar.tapered_opening_angle(f, connection, two_spin, single_spin),
            _tapered_opening_angle(f, connection, two_spin, single_spin),
            rel_tol=2e-12,
            abs_tol=2e-12,
        )
    assert scalar.tapered_opening_angle(0.0, connection, two_spin, single_spin) == two_spin
    assert scalar.tapered_opening_angle(connection, connection, two_spin, single_spin) == single_spin


def test_beta_taper_torch_gradients_and_connection():
    import torch

    from torchwave.approximants.pnr_attachment import tapered_opening_angle

    inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (0.03, 0.08, 1.1, 0.7)]
    actual = tapered_opening_angle(*inputs)
    gradients = torch.autograd.grad(actual, inputs)
    assert math.isclose(actual.item(), _tapered_opening_angle(0.03, 0.08, 1.1, 0.7), rel_tol=2e-12)
    assert all(torch.isfinite(gradient) for gradient in gradients)
    at_join = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
    joined = tapered_opening_angle(at_join, *inputs[1:])
    assert torch.autograd.grad(joined, at_join)[0].item() == pytest.approx(0.0, abs=2e-12)


def test_beta_taper_jax_jit_gradients_and_connection():
    import jax
    import jax.numpy as jnp

    from jaxwave.approximants.pnr_attachment import tapered_opening_angle

    values = jnp.asarray((0.03, 0.08, 1.1, 0.7), dtype=jnp.float64)
    function = jax.jit(lambda x: tapered_opening_angle(x[0], x[1], x[2], x[3]))
    actual = function(values)
    gradients = jax.jit(jax.grad(lambda x: tapered_opening_angle(x[0], x[1], x[2], x[3])))(values)
    assert math.isclose(float(actual), _tapered_opening_angle(*map(float, values)), rel_tol=2e-12)
    assert bool(jnp.all(jnp.isfinite(gradients)))
    at_join = jnp.asarray((0.08, 0.08, 1.1, 0.7), dtype=jnp.float64)
    assert float(jax.grad(lambda f: tapered_opening_angle(f, *at_join[1:]))(at_join[0])) == pytest.approx(0.0, abs=2e-12)


def test_optimal_emission_direction_scalar_uses_corrected_denominator():
    expected = _optimal_emission_direction_beta(*EMISSION_ARGS)
    actual = scalar.optimal_emission_direction_beta(*EMISSION_ARGS)
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12)
    assert actual == pytest.approx(0.7899719998388846, rel=2e-14, abs=2e-14)


def test_optimal_emission_direction_torch_value_and_gradients():
    import torch

    from torchwave.approximants.pnr_attachment import optimal_emission_direction_beta

    inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in EMISSION_ARGS]
    actual = optimal_emission_direction_beta(*inputs)
    gradients = torch.autograd.grad(actual, inputs)
    assert math.isclose(actual.item(), _optimal_emission_direction_beta(*EMISSION_ARGS), rel_tol=2e-12)
    assert all(torch.isfinite(gradient) for gradient in gradients)


def test_optimal_emission_direction_jax_jit_value_and_gradients():
    import jax
    import jax.numpy as jnp

    from jaxwave.approximants.pnr_attachment import optimal_emission_direction_beta

    values = jnp.asarray(EMISSION_ARGS, dtype=jnp.float64)
    function = jax.jit(lambda x: optimal_emission_direction_beta(*x))
    actual = function(values)
    gradients = jax.jit(jax.grad(lambda x: optimal_emission_direction_beta(*x)))(values)
    assert math.isclose(float(actual), _optimal_emission_direction_beta(*EMISSION_ARGS), rel_tol=2e-12)
    assert bool(jnp.all(jnp.isfinite(gradients)))


@pytest.mark.parametrize("v,perpendicular,parallel", [(0.27, 0.0, 0.0), (0.0, 0.0, 0.4)])
def test_optimal_emission_direction_degenerate_spin_and_frequency(
    v, perpendicular, parallel,
):
    import jax
    import jax.numpy as jnp
    import torch

    from jaxwave.approximants import pnr_attachment as jax_attachment
    from torchwave.approximants import pnr_attachment as torch_attachment

    args = (v, 0.8, 0.22, 0.35, perpendicular, parallel)
    assert scalar.optimal_emission_direction_beta(*args) == pytest.approx(0.8, abs=2e-14)

    torch_args = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in args]
    torch_value = torch_attachment.optimal_emission_direction_beta(*torch_args)
    torch_grad = torch.autograd.grad(torch_value, torch_args)
    assert torch_value.item() == pytest.approx(0.8, abs=2e-14)
    assert all(torch.isfinite(gradient) for gradient in torch_grad)

    jax_args = jnp.asarray(args, dtype=jnp.float64)
    jax_value = jax.jit(lambda values: jax_attachment.optimal_emission_direction_beta(*values))(jax_args)
    jax_grad = jax.jit(jax.grad(lambda values: jax_attachment.optimal_emission_direction_beta(*values)))(jax_args)
    assert float(jax_value) == pytest.approx(0.8, abs=2e-14)
    assert bool(jnp.all(jnp.isfinite(jax_grad)))
