import torch
import pytest

from torchwave.approximants.xhm_intermediate_amplitude_clean import (
    evaluate_intermediate_amplitude,
    evaluate_intermediate_inverse_polynomial,
    solve_intermediate_inverse_polynomial,
    solve_intermediate_normalized_amplitude,
)


def test_polynomial_constraints_and_c1_endpoints():
    expected = torch.tensor([2.0, .3, -.2, .1, .04, -.01], dtype=torch.float64)
    def p(u): return evaluate_intermediate_inverse_polynomial(torch.tensor(u, dtype=torch.float64), expected)
    values = [p(0), expected[1], p(1/3), p(2/3), p(1), sum(i * expected[i] for i in range(1, 6))]
    solved = solve_intermediate_inverse_polynomial(values[0], values[1], values[2:4], values[4], values[5])
    assert torch.allclose(solved, expected, atol=3e-12, rtol=3e-12)
    frequency = torch.tensor([0.0, 3.0], dtype=torch.float64)
    joined = evaluate_intermediate_amplitude(frequency, 0.0, 3.0, 2.0, solved)
    assert torch.allclose(joined, torch.tensor([1.0, 2.0 / p(1)], dtype=torch.float64), atol=3e-12, rtol=3e-12)
    f = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    evaluate_intermediate_amplitude(f, 0.0, 3.0, 2.0, expected).backward()
    assert torch.allclose(f.grad, torch.tensor(-0.05, dtype=torch.float64), atol=1e-12)


def test_batching_gradients_and_physical_units():
    start = torch.tensor([2.0, 2.2], dtype=torch.float64, requires_grad=True)
    coeff = solve_intermediate_inverse_polynomial(start, torch.ones(2, dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64), 2.0, 1.0)
    amp = evaluate_intermediate_amplitude(torch.tensor([10., 15.], dtype=torch.float64), 5., 20., 4., coeff)
    amp.sum().backward()
    assert amp.shape == (2,) and torch.isfinite(start.grad).all()


def test_bad_shapes_and_poles_rejected():
    with pytest.raises(ValueError, match="trailing shape"):
        solve_intermediate_inverse_polynomial(1., 1., [1.], 1., 1.)
    with pytest.raises(ValueError, match="(pole|certify)"):
        evaluate_intermediate_inverse_polynomial(torch.tensor(.5), torch.tensor([-1., 0., 0., 0., 0., 1.]))
    # A negative pocket on the right must not be lost while subdividing a
    # separate unresolved pocket on the left.
    with pytest.raises(ValueError, match="certify"):
        evaluate_intermediate_inverse_polynomial(torch.tensor(.2), torch.tensor([1., -20., 100., -160., 80., -10.]))
    # A tangent root has no sign change and is therefore caught by the
    # interval certificate as well.
    root = 0.5005
    with pytest.raises(ValueError, match="certify"):
        evaluate_intermediate_inverse_polynomial(torch.tensor(.1), torch.tensor([root * root, -2 * root, 1., 0., 0., 0.]))
    with pytest.raises(ValueError, match="f_end"):
        evaluate_intermediate_amplitude(1., 2., 2., 1., torch.ones(6))
    with pytest.raises(ValueError, match="join interval"):
        evaluate_intermediate_amplitude(21., 5., 20., 1., torch.ones(6))


def test_tight_pole_certification_and_broadcasting():
    # Tangent zero strictly between sample locations.
    root = 0.5005
    tangent = torch.tensor([root * root, -2 * root, 1., 0., 0., 0.], dtype=torch.float64)
    with pytest.raises(ValueError, match="certify"):
        evaluate_intermediate_inverse_polynomial(torch.tensor([.1, .9]), tangent)
    coeff = torch.tensor([2., .3, -.2, .1, .04, -.01], dtype=torch.float64).reshape(1, 1, 6).expand(2, 1, 6)
    out = evaluate_intermediate_amplitude(torch.tensor([.1, .5, .9], dtype=torch.float64), 0., 1., 2., coeff)
    assert out.shape == (2, 3)
    # Positive polynomial whose power-basis Bernstein certificate needs a split.
    positive = torch.tensor([.26, -1., 1., 0., 0., 0.], dtype=torch.float64)
    assert evaluate_intermediate_inverse_polynomial(.5, positive).item() == pytest.approx(.01, abs=1e-12)


def test_normalized_amplitude_wrapper_and_gradcheck():
    coeff = solve_intermediate_normalized_amplitude(
        1.0, .2, [1.1, 1.2], 1.3, -.1, 10.0, 20.0
    )
    assert torch.allclose(
        evaluate_intermediate_amplitude(torch.tensor([10., 20.]), 10., 20., 3., coeff),
        torch.tensor([3., 3. * 1.3], dtype=torch.float64), atol=1e-12, rtol=1e-12,
    )
    for f, expected in ((10., .6), (20., -.3)):
        point = torch.tensor(f, dtype=torch.float64, requires_grad=True)
        evaluate_intermediate_amplitude(point, 10., 20., 3., coeff).backward()
        assert point.grad.item() == pytest.approx(expected, abs=1e-11)
    variable = coeff.detach().requires_grad_()
    frequencies = torch.tensor([12., 16.], dtype=torch.float64)
    def fn(c):
        return evaluate_intermediate_amplitude(frequencies, 10., 20., 3., c)
    assert torch.autograd.gradcheck(fn, (variable,), eps=1e-6, atol=1e-5, rtol=1e-5)


def test_gradcheck_through_normalized_solver_inputs():
    def fn(start, interior):
        c = solve_intermediate_normalized_amplitude(start, .1, interior, 1.3, -.1, 10., 20.)
        return evaluate_intermediate_amplitude(torch.tensor([12.], dtype=torch.float64), 10., 20., 3., c)
    start = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    interior = torch.tensor([1.1, 1.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (start, interior), eps=1e-6, atol=1e-5, rtol=1e-5)
