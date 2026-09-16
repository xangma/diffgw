"""Primary-equation limits and mathematical checks; no external model oracle."""

from decimal import Decimal, localcontext
import math

import pytest
import torch

from torchwave.approximants.eob.remnant import (
    _kerr_isco,
    eob_final_mass_fraction,
    hofmann_final_spin,
)


ORDERS = ((1, 2), (3, 3), (3, 4))


def real(value):
    return torch.tensor(value, dtype=torch.float64)


def evaluate(order, *arguments):
    if order is None:
        return eob_final_mass_fraction(*arguments)
    return hofmann_final_spin(*arguments, fit_order=order)


def isco_reference(spin):
    # Independent root of the marginal-stability equation. Decimal arithmetic
    # resolves the root degeneracy close to the prograde extremal endpoint.
    with localcontext() as context:
        context.prec = 75
        a = Decimal.from_float(float(spin))
        low, high = Decimal(1), Decimal(9)
        for _ in range(240):
            radius = (low+high)/2
            residual = radius*radius - 6*radius + 8*a*radius.sqrt() - 3*a*a
            if residual > 0:
                high = radius
            else:
                low = radius
        radius = Decimal(1) if a == 1 else (low+high)/2
        energy = (1-2/(3*radius)).sqrt()
        angular = 2/(3*Decimal(3).sqrt())*(1+2*(3*radius-2).sqrt())
        return tuple(float(value) for value in (radius, energy, angular))


def test_signed_isco_matches_independent_marginal_stability_roots():
    spin = real([-1., -1+1e-12, -.7, -1e-12, 0., 1e-12, .8, 1-1e-12, 1.])
    expected = real([isco_reference(value) for value in spin])
    actual = torch.stack(_kerr_isco(spin), -1)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    assert torch.all(actual[:, 0].diff() < 0)
    # L denotes the positive orbital direction, including retrograde Kerr.
    assert torch.all(actual[:, 2] > 0)


def test_isco_zero_and_retrograde_endpoint_have_physical_derivatives():
    spin = real([-1., 0., .8]).requires_grad_(True)
    radius = _kerr_isco(spin)[0]
    gradient = torch.autograd.grad(radius.sum(), spin)[0]
    torch.testing.assert_close(gradient[:2], real([-45/16, -4*math.sqrt(2/3)]),
                               rtol=2e-15, atol=2e-15)
    interior = real([-.9, 0., .7]).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a: torch.stack(_kerr_isco(a)), (interior,))
    assert torch.autograd.gradgradcheck(lambda a: torch.stack(_kerr_isco(a)), (interior,))
    extremal = real(1.).requires_grad_(True)
    assert torch.isneginf(torch.autograd.grad(_kerr_isco(extremal)[0], extremal)[0])


@pytest.mark.parametrize("order", ORDERS)
def test_equal_mass_nonspinning_constraint_is_imposed(order):
    actual = hofmann_final_spin(30., 30., 0., 0., fit_order=order)
    torch.testing.assert_close(actual, real(.68646), rtol=0, atol=3e-16)


def test_equal_mass_radiation_is_the_printed_rational_fit():
    chi1, chi2 = real([-.9, 0., .8]), real([-.3, 0., 1.])
    expected = 1 - (.00258 - .0773/((chi1+chi2)/2 - 1.6939))
    torch.testing.assert_close(eob_final_mass_fraction(1., 1., chi1, chi2), expected,
                               rtol=0, atol=2e-16)


@pytest.mark.parametrize("order", (None, *ORDERS))
def test_test_particle_limit_has_the_published_first_order_coefficient(order):
    # Hofmann Eq. (1) and the nu->0 limit of Ossokine Eq. (3.5).
    # Use a decreasing sequence to distinguish the first-order coefficient
    # from merely returning the limiting primary spin/mass.
    for spin in (-.8, 0., .8):
        _, energy, angular = isco_reference(spin)
        target = 1-energy if order is None else angular - 2*spin*energy
        errors = []
        for q in (1e-3, 1e-4, 1e-5):
            nu = q/(1+q)**2
            result = evaluate(order, 1., q, spin, -.35).item()
            coefficient = (1-result)/nu if order is None else (result-spin)/nu
            errors.append(abs(coefficient-target))
        assert errors[2] < errors[1]/8 < errors[0]/64
        assert errors[2] < 8e-4


@pytest.mark.parametrize("order", (None, *ORDERS))
def test_parameter_batches_exchange_symmetry_and_mass_units(order):
    m1, m2 = real([[10.], [100.]]), real([10., 3., 7.])
    s1, s2 = real([-.8, 0., .7]), real(.2)
    actual = evaluate(order, m1, m2, s1, s2)
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, evaluate(order, m2, m1, s2, s1), rtol=2e-15, atol=2e-15)
    for factor in (1e-250, 7., 1e250):
        torch.testing.assert_close(actual, evaluate(order, m1*factor, m2*factor, s1, s2),
                                   rtol=2e-15, atol=2e-15)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(actual[i, j], evaluate(order, m1[i, 0], m2[j], s1[j], s2))


@pytest.mark.parametrize("order", (None, *ORDERS))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dtype_execution_modes_and_endpoint_component_spins(order, dtype):
    spin = torch.tensor([-1., 0., 1.], dtype=dtype)
    actual = evaluate(order, 10., 1., spin, spin)
    assert actual.dtype == dtype
    assert torch.all(torch.isfinite(actual))
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            torch.testing.assert_close(evaluate(order, 10., 1., spin, spin), actual)
    promoted = evaluate(order, real(10.), 1., spin, spin)
    assert promoted.dtype == torch.float64


@pytest.mark.parametrize("order", (None, *ORDERS))
def test_all_intrinsic_gradients_including_zero_spin_and_equal_mass(order):
    parameters = real([[30., 12., -.6, .7], [20., 20., 0., 0.]]).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: evaluate(order, *p.unbind(-1)), (parameters,),
                                    atol=2e-7, rtol=3e-5)


@pytest.mark.parametrize("order", ORDERS)
def test_negative_final_spin_keeps_its_sign(order):
    result = hofmann_final_spin(100., 1., -.9, -.5, fit_order=order)
    assert -1 < result < 0


@pytest.mark.parametrize("order", (None, *ORDERS))
@pytest.mark.parametrize("arguments", [
    (0., 2., 0., 0.), (-1., 2., 0., 0.), (1., float("inf"), 0., 0.),
    (1., 2., float("nan"), 0.), (1., 2., -1.01, 0.), (1., 2., 0., 1.01),
])
def test_invalid_physical_inputs_fail(order, arguments):
    with pytest.raises(ValueError):
        evaluate(order, *arguments)


def test_fit_selection_is_required_and_validated():
    with pytest.raises(TypeError):
        hofmann_final_spin(1., 1., 0., 0.)
    for order in (None, "3,4", [3, 4], (2, 3), (True, 2), (3., 4.)):
        with pytest.raises(ValueError, match="fit_order"):
            hofmann_final_spin(1., 1., 0., 0., fit_order=order)


def test_unsupported_tensor_and_scalar_types_fail():
    for mass in (torch.tensor(1), torch.tensor(1., dtype=torch.float16), 1+1j, True):
        with pytest.raises(TypeError):
            eob_final_mass_fraction(mass, 1., 0., 0.)


@pytest.mark.parametrize("order", (None, *ORDERS))
@pytest.mark.parametrize("device", ("mps", "cuda"))
def test_accelerator_device_and_gradients(order, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    dtype = torch.float32 if device == "mps" else torch.float64
    spins = torch.tensor([-.8, 0., .8], device=device, dtype=dtype, requires_grad=True)
    result = evaluate(order, 10., 3., spins, .2)
    assert result.device == spins.device
    assert torch.all(torch.isfinite(torch.autograd.grad(result.sum(), spins)[0]))
    torch.testing.assert_close(result.detach().cpu(), evaluate(order, 10., 3., spins.detach().cpu(), .2),
                               rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError, match="device"):
        evaluate(order, torch.tensor(10., dtype=dtype), 3., spins, .2)
