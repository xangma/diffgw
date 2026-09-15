import pytest
import torch

from scripts.nr_complex_reduced_basis import ComplexReducedBasisModel, fit_complex_basis


def _analytic(q):
    """Rank-two complex coordinates in the module's intrinsic coordinate."""
    x = 4 * q / (1 + q).square() - 1
    c1 = (0.7 + 0.2j) + (0.4 - 0.3j) * x + (0.15 + 0.1j) * x.square()
    c2 = (-0.2 + 0.5j) + (0.1 + 0.25j) * x
    basis = torch.tensor(
        [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.complex128
    )
    return torch.stack((c1, c2), dim=-1) @ basis


def test_complex_basis_recovers_unseen_low_rank_function_and_shapes():
    q = torch.linspace(1.0, 8.0, 9, dtype=torch.float64)
    values = _analytic(q).reshape(q.numel(), 2, 3)
    model = fit_complex_basis(q, values, rank=2, degree=2)

    query = torch.tensor([1.5, 3.7, 7.2], dtype=torch.float64)
    got = model.predict(query)
    expected = _analytic(query).reshape(-1, 2, 3)
    assert got.shape == (3, 2, 3)
    assert torch.allclose(got, expected, atol=2e-12, rtol=2e-12)
    assert model.predict(query[0]).shape == (2, 3)
    assert torch.allclose(model(query), got)


def test_batch_equivalence_and_jvp_match_centered_difference():
    q = torch.linspace(1.0, 7.0, 9, dtype=torch.float64)
    model = fit_complex_basis(q, _analytic(q).reshape(-1, 2, 3), rank=2, degree=2)
    query = torch.tensor(3.3, dtype=torch.float64, requires_grad=True)
    value = model.predict(query)
    tangent = torch.ones_like(query)
    _, jvp = torch.func.jvp(model.predict, (query,), (tangent,))
    eps = 1.0e-5
    fd = (model.predict(query.detach() + eps) - model.predict(query.detach() - eps)) / (2 * eps)
    assert torch.allclose(jvp, fd, atol=2e-8, rtol=2e-7)
    batched = model.predict(torch.stack((query, query + 0.2)))
    assert torch.allclose(batched[0], value)


def test_training_only_and_serialization_round_trip():
    q = torch.linspace(1.0, 6.0, 8, dtype=torch.float64)
    train = _analytic(q).reshape(-1, 2, 3)
    model = fit_complex_basis(q, train, rank=2, degree=2)
    reference = model.predict(torch.tensor([1.2, 4.2], dtype=torch.float64))
    heldout = torch.randn(4, 2, 3, dtype=torch.complex128)
    heldout.mul_(1000)
    assert torch.allclose(model.predict(torch.tensor([1.2, 4.2], dtype=torch.float64)), reference)
    restored = ComplexReducedBasisModel.from_dict(model.to_dict())
    assert torch.allclose(restored.predict(torch.tensor([1.2, 4.2], dtype=torch.float64)), reference)


def test_invalid_data_rank_weights_and_extrapolation_are_guarded():
    q = torch.linspace(1.0, 4.0, 5, dtype=torch.float64)
    values = _analytic(q).reshape(-1, 2, 3)
    model = fit_complex_basis(q, values, rank=2, degree=2)
    with pytest.raises(ValueError, match="outside"):
        model.predict(torch.tensor(4.1, dtype=torch.float64))
    assert model.predict(torch.tensor(4.1, dtype=torch.float64), allow_extrapolation=True).shape == (2, 3)
    with pytest.raises(ValueError, match="q >= 1"):
        fit_complex_basis(torch.tensor([0.9, 1.2, 1.4], dtype=torch.float64), values[:3], rank=1, degree=1)
    with pytest.raises(ValueError, match="independent"):
        fit_complex_basis(torch.ones(5, dtype=torch.float64), values, rank=1, degree=1)
    with pytest.raises(ValueError, match="rank"):
        fit_complex_basis(q, values, rank=3, degree=2)
    with pytest.raises(ValueError, match="weights"):
        fit_complex_basis(q, values, rank=1, degree=1, weights=torch.zeros(5, dtype=torch.float64))
    bad = values.clone()
    bad[0, 0, 0] = complex(float("nan"), 0)
    with pytest.raises(ValueError, match="finite"):
        fit_complex_basis(q, bad, rank=1, degree=1)
